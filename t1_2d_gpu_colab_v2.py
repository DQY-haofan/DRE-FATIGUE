"""
T1 2D Verification - FIXED VERSION with diagnostics
====================================================
Fixes from v1:
1. Use float64 everywhere (rule out precision issues)
2. Add diagnostics after each solve stage  
3. Sanity check: uniform tension test on 2x2 mesh FIRST
4. Fix potential PyTorch indexing issues in BC application

Copy entire file into Colab cell, run with GPU.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time, json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64   # <-- CHANGED: use float64
print(f"Device: {device}, dtype: {dtype}")

# --- Physics ---
E, nu = 210.0, 0.3
Gc = 2.7e-3
k_res = 1e-7
lam = E*nu/((1+nu)*(1-2*nu))
mu  = E/(2*(1+nu))

# --- Gauss quadrature ---
gp = 1.0/3**0.5
GP_XI  = torch.tensor([-gp, gp, gp, -gp], device=device, dtype=dtype)
GP_ETA = torch.tensor([-gp, -gp, gp, gp], device=device, dtype=dtype)
GP_W   = torch.ones(4, device=device, dtype=dtype)

def shape_at_gp(xi, eta):
    N = 0.25 * torch.stack([
        (1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
    dN = 0.25 * torch.stack([
        torch.stack([-(1-eta),  (1-eta),  (1+eta), -(1+eta)]),
        torch.stack([-(1-xi),  -(1+xi),  (1+xi),   (1-xi) ])])
    return N, dN

N_gps, dN_gps = [], []
for q in range(4):
    N_q, dN_q = shape_at_gp(GP_XI[q], GP_ETA[q])
    N_gps.append(N_q); dN_gps.append(dN_q)


# =================================================================
# MESH
# =================================================================
def create_mesh(nx, ny):
    h = 1.0 / max(nx, ny)
    x = torch.linspace(0, 1, nx+1, device=device, dtype=dtype)
    y = torch.linspace(0, 1, ny+1, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    nodes = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    n_nodes = nodes.shape[0]

    j_idx = torch.arange(ny, device=device).unsqueeze(1).expand(ny, nx).reshape(-1)
    i_idx = torch.arange(nx, device=device).unsqueeze(0).expand(ny, nx).reshape(-1)
    n0 = j_idx*(nx+1) + i_idx
    elements = torch.stack([n0, n0+1, n0+(nx+1)+1, n0+(nx+1)], dim=1).long()

    # Pre-crack
    crack_mask = (torch.abs(nodes[:,1] - 0.5) < h*0.6) & (nodes[:,0] <= 0.5 + h*0.6)
    d_init = torch.zeros(n_nodes, device=device, dtype=dtype)
    d_init[crack_mask] = 1.0

    bot = nodes[:,1] < 1e-10
    top = nodes[:,1] > 1.0 - 1e-10

    return nodes, elements, d_init, n_nodes, bot, top, h


# =================================================================
# CORE FEM ROUTINES
# =================================================================
def get_dN_dx(dN_q, xe):
    """Compute dN/dx, detJ for all elements. xe: (n_elem,4,2)"""
    J = torch.einsum('ij,ejk->eik', dN_q, xe)
    detJ = J[:,0,0]*J[:,1,1] - J[:,0,1]*J[:,1,0]
    inv_detJ = 1.0 / detJ
    invJ = torch.zeros_like(J)
    invJ[:,0,0] =  J[:,1,1] * inv_detJ
    invJ[:,0,1] = -J[:,0,1] * inv_detJ
    invJ[:,1,0] = -J[:,1,0] * inv_detJ
    invJ[:,1,1] =  J[:,0,0] * inv_detJ
    dN_dx = torch.einsum('eij,jk->eik', invJ, dN_q)
    return dN_dx, detJ

def build_B(dN_dx, n_elem):
    """Strain-displacement matrix B: (n_elem, 3, 8)"""
    B = torch.zeros(n_elem, 3, 8, device=device, dtype=dtype)
    for i in range(4):
        B[:, 0, 2*i]   = dN_dx[:, 0, i]   # exx <- ux
        B[:, 1, 2*i+1] = dN_dx[:, 1, i]   # eyy <- uy
        B[:, 2, 2*i]   = dN_dx[:, 1, i]   # 2exy <- ux
        B[:, 2, 2*i+1] = dN_dx[:, 0, i]   # 2exy <- uy
    return B

def elem_dofs(elements):
    """DOF indices: (n_elem, 8) = [ux0,uy0,ux1,uy1,...]"""
    n_elem = elements.shape[0]
    d = torch.stack([2*elements, 2*elements+1], dim=-1)  # (n_elem,4,2)
    return d.reshape(n_elem, 8)


def solve_u(nodes, elements, d_field, bot, top, u_bar, n_nodes, debug=False):
    """Displacement sub-problem with penalty BCs."""
    n_elem = elements.shape[0]
    n_dof = 2 * n_nodes
    xe = nodes[elements]       # (n_elem, 4, 2)
    de = d_field[elements]     # (n_elem, 4)
    dofs = elem_dofs(elements) # (n_elem, 8)

    D0 = torch.tensor([
        [lam+2*mu, lam, 0],
        [lam, lam+2*mu, 0],
        [0, 0, mu]], device=device, dtype=dtype)

    Ke = torch.zeros(n_elem, 8, 8, device=device, dtype=dtype)
    for q in range(4):
        dN_dx, detJ = get_dN_dx(dN_gps[q], xe)
        B = build_B(dN_dx, n_elem)
        d_gp = (N_gps[q].unsqueeze(0) * de).sum(dim=1)  # (n_elem,)
        g_val = (1 - d_gp)**2 + k_res
        BtDB = torch.einsum('eji,jk,ekl->eil', B, D0, B)
        Ke += (g_val * detJ * GP_W[q]).unsqueeze(-1).unsqueeze(-1) * BtDB

    # Sparse -> Dense assembly
    row = dofs.unsqueeze(2).expand(-1,8,8).reshape(-1)
    col = dofs.unsqueeze(1).expand(-1,8,8).reshape(-1)
    K = torch.zeros(n_dof, n_dof, device=device, dtype=dtype)
    K.index_put_((row, col), Ke.reshape(-1), accumulate=True)

    f = torch.zeros(n_dof, device=device, dtype=dtype)

    if debug:
        print(f"    [solve_u] K: max={K.abs().max().item():.4e}, "
              f"diag_max={K.diag().max().item():.4e}, nnz(>1e-15)={(K.abs()>1e-15).sum().item()}")

    # Penalty BCs
    penalty = K.diag().max().item() * 1e7
    
    bot_idx = torch.where(bot)[0]
    top_idx = torch.where(top)[0]
    
    # Bottom: ux=0, uy=0
    for dof_offset in [0, 1]:
        dof_ids = 2*bot_idx + dof_offset
        K[dof_ids, :] = 0
        K[dof_ids, dof_ids] = penalty
        f[dof_ids] = 0.0
    
    # Top: uy = u_bar (leave ux free)
    dof_ids = 2*top_idx + 1
    K[dof_ids, :] = 0
    K[dof_ids, dof_ids] = penalty
    f[dof_ids] = penalty * u_bar

    u = torch.linalg.solve(K, f)

    if debug:
        print(f"    [solve_u] u: max={u.abs().max().item():.6e}, "
              f"uy_top={u[2*top_idx[0]+1].item():.6e} (should be {u_bar:.6e})")
    return u


def compute_H(nodes, elements, u_field, H_prev, n_nodes, debug=False):
    """H = max(H_prev, psi_e+) at nodes."""
    n_elem = elements.shape[0]
    xe = nodes[elements]
    dofs = elem_dofs(elements)
    ue = u_field[dofs]  # (n_elem, 8)

    if debug:
        print(f"    [compute_H] max|ue|={ue.abs().max().item():.6e}")

    psi_accum = torch.zeros(n_nodes, device=device, dtype=dtype)
    weight = torch.zeros(n_nodes, device=device, dtype=dtype)

    for q in range(4):
        dN_dx, detJ = get_dN_dx(dN_gps[q], xe)
        B = build_B(dN_dx, n_elem)
        eps = torch.einsum('eij,ej->ei', B, ue)  # (n_elem, 3)

        if debug and q == 0:
            print(f"    [compute_H] eps at GP0: max|exx|={eps[:,0].abs().max().item():.6e}, "
                  f"max|eyy|={eps[:,1].abs().max().item():.6e}")

        # Spectral split
        exx, eyy = eps[:,0], eps[:,1]
        gxy = eps[:,2] * 0.5  # engineering -> tensor shear
        trace = exx + eyy
        det = exx*eyy - gxy**2
        disc = torch.clamp(trace**2 - 4*det, min=0)
        e1 = 0.5*(trace + torch.sqrt(disc))
        e2 = 0.5*(trace - torch.sqrt(disc))

        trace_p = torch.clamp(trace, min=0)
        e1_p = torch.clamp(e1, min=0)
        e2_p = torch.clamp(e2, min=0)
        psi_plus = 0.5*lam*trace_p**2 + mu*(e1_p**2 + e2_p**2)

        if debug and q == 0:
            print(f"    [compute_H] psi+ at GP0: max={psi_plus.max().item():.6e}, "
                  f"mean={psi_plus.mean().item():.6e}")

        # Scatter to nodes (weighted average)
        for i in range(4):
            w_i = N_gps[q][i].abs()  # shape function weight
            psi_accum.scatter_add_(0, elements[:,i], psi_plus * w_i * detJ)
            weight.scatter_add_(0, elements[:,i], w_i * detJ * torch.ones(n_elem, device=device, dtype=dtype))

    psi_nodal = psi_accum / weight.clamp(min=1e-30)
    H_new = torch.maximum(H_prev, psi_nodal)

    if debug:
        print(f"    [compute_H] psi_nodal: max={psi_nodal.max().item():.6e}, "
              f"H: max={H_new.max().item():.6e}")
    return H_new


def solve_d(nodes, elements, H_field, d_prev, n_nodes, ell_val):
    """Phase-field sub-problem."""
    n_elem = elements.shape[0]
    xe = nodes[elements]
    He = H_field[elements]

    Ke = torch.zeros(n_elem, 4, 4, device=device, dtype=dtype)
    fe = torch.zeros(n_elem, 4, device=device, dtype=dtype)

    for q in range(4):
        dN_dx, detJ = get_dN_dx(dN_gps[q], xe)
        H_gp = (N_gps[q].unsqueeze(0) * He).sum(dim=1)
        N_outer = torch.einsum('i,j->ij', N_gps[q], N_gps[q])
        grad_term = Gc * ell_val * torch.einsum('eji,ejk->eik', dN_dx, dN_dx)
        mass_coeff = Gc/(2*ell_val) + 2*H_gp
        mass_term = mass_coeff.unsqueeze(-1).unsqueeze(-1) * N_outer.unsqueeze(0)
        w_detJ = (GP_W[q] * detJ).unsqueeze(-1).unsqueeze(-1)
        Ke += (grad_term + mass_term) * w_detJ
        fe += (2 * H_gp * GP_W[q] * detJ).unsqueeze(-1) * N_gps[q].unsqueeze(0)

    # Assembly
    row = elements.unsqueeze(2).expand(-1,4,4).reshape(-1)
    col = elements.unsqueeze(1).expand(-1,4,4).reshape(-1)
    K_d = torch.zeros(n_nodes, n_nodes, device=device, dtype=dtype)
    K_d.index_put_((row, col), Ke.reshape(-1), accumulate=True)

    rhs = torch.zeros(n_nodes, device=device, dtype=dtype)
    rhs.scatter_add_(0, elements.reshape(-1), fe.reshape(-1))

    d = torch.linalg.solve(K_d, rhs)
    d = torch.clamp(torch.maximum(d, d_prev), 0, 1)
    return d


def staggered_step(nodes, elements, u_bar, d_prev, H_prev, n_nodes,
                   bot, top, ell_val, max_stagger=20, tol=1e-6, debug=False):
    d = d_prev.clone()
    H = H_prev.clone()
    for it in range(max_stagger):
        d_old = d.clone()
        u = solve_u(nodes, elements, d, bot, top, u_bar, n_nodes, debug=(debug and it==0))
        H = compute_H(nodes, elements, u, H, n_nodes, debug=(debug and it==0))
        d = solve_d(nodes, elements, H, d_prev, n_nodes, ell_val)
        dd = (d - d_old).abs().max().item()
        if debug:
            print(f"    stagger iter {it+1}: max|Δd|={dd:.2e}")
        if dd < tol:
            break
    return u, d, H


# =================================================================
# SANITY CHECK: 2x2 mesh, uniform tension, no crack
# =================================================================
def sanity_check():
    print("="*60)
    print("SANITY CHECK: 2x2 mesh, uniform tension")
    print("="*60)

    nx, ny = 4, 4
    h_test = 1.0/4
    x = torch.linspace(0,1,nx+1, device=device, dtype=dtype)
    y = torch.linspace(0,1,ny+1, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    nodes = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    n_nodes = nodes.shape[0]

    j_idx = torch.arange(ny, device=device).unsqueeze(1).expand(ny,nx).reshape(-1)
    i_idx = torch.arange(nx, device=device).unsqueeze(0).expand(ny,nx).reshape(-1)
    n0 = j_idx*(nx+1)+i_idx
    elements = torch.stack([n0,n0+1,n0+(nx+1)+1,n0+(nx+1)], dim=1).long()

    d_field = torch.zeros(n_nodes, device=device, dtype=dtype)  # no damage
    bot = nodes[:,1] < 1e-10
    top = nodes[:,1] > 1.0-1e-10
    H_prev = torch.zeros(n_nodes, device=device, dtype=dtype)

    u_test = 0.01
    print(f"  Applying uy={u_test} on top, fixed bottom, no crack")

    u = solve_u(nodes, elements, d_field, bot, top, u_test, n_nodes, debug=True)
    H = compute_H(nodes, elements, u, H_prev, n_nodes, debug=True)

    # Expected: uniform eyy = u_test/L = 0.01, exx from Poisson
    # For plane strain: sigma_yy = (lam+2*mu)*eyy, sigma_xx = lam*eyy
    # psi = 0.5*(lam+2*mu)*eyy^2 + 0.5*lam*eyy^2 ... no, more carefully:
    # eps = [exx, eyy, 0], with exx from sigma_xx = 0 constraint:
    # Actually in plane strain with free sides: exx ≠ 0 (not plane stress!)
    # For fixed bottom, free sides, prescribed top: 
    # Rough check: eyy ≈ u_test, exx ≈ 0 (if sides free)
    # psi_e+ ≈ 0.5*(lam+2*mu)*eyy^2 ≈ 0.5*282.7*0.01^2 ≈ 0.0141

    expected_psi = 0.5 * (lam + 2*mu) * u_test**2
    print(f"\n  Expected psi_e+ ≈ {expected_psi:.6f}")
    print(f"  Actual max(H)   = {H.max().item():.6f}")

    if H.max().item() > 0.001:
        print("  ✅ SANITY CHECK PASSED")
        return True
    else:
        print("  ❌ SANITY CHECK FAILED - debugging needed")
        return False


# =================================================================
# HISTORY TRANSFER SCHEMES
# =================================================================
def transfer_coarse_grid(H, nx_mesh, ny_mesh, n_storage):
    """Store on n_storage x n_storage grid, bilinear interp back."""
    H_grid = H.reshape(ny_mesh+1, nx_mesh+1).unsqueeze(0).unsqueeze(0)
    # Down-sample
    H_coarse = F.interpolate(H_grid, size=(n_storage, n_storage), mode='bilinear', align_corners=True)
    # Up-sample back
    H_recon = F.interpolate(H_coarse, size=(ny_mesh+1, nx_mesh+1), mode='bilinear', align_corners=True)
    return torch.clamp(H_recon.reshape(-1), min=0)

def transfer_rbf(H, nodes, n_centers):
    """RBF regression with n_centers."""
    n_nodes = nodes.shape[0]
    idx = torch.linspace(0, n_nodes-1, n_centers, device=device).long()
    centers = nodes[idx]
    H_c = H[idx]
    sigma = 1.0 / n_centers**0.5
    diff = nodes.unsqueeze(1) - centers.unsqueeze(0)
    Phi = torch.exp(-(diff**2).sum(-1) / (2*sigma**2))
    Phi_c = Phi[idx]
    w = torch.linalg.solve(Phi_c + 1e-6*torch.eye(n_centers, device=device, dtype=dtype), H_c)
    return torch.clamp(Phi @ w, min=0)

def transfer_random(H, nodes, nx_mesh, ny_mesh, n_col, seed):
    """Store at random points, smooth reconstruction."""
    gen = torch.Generator(device=device).manual_seed(seed)
    n_nodes = nodes.shape[0]
    idx = torch.randperm(n_nodes, generator=gen, device=device)[:n_col]
    
    # Interpolate from random subset back to full grid via RBF
    centers = nodes[idx]
    H_c = H[idx]
    sigma = 2.0 / n_col**0.5
    diff = nodes.unsqueeze(1) - centers.unsqueeze(0)
    Phi = torch.exp(-(diff**2).sum(-1) / (2*sigma**2))
    Phi_c = Phi[idx]
    w = torch.linalg.solve(Phi_c + 1e-4*torch.eye(n_col, device=device, dtype=dtype), H_c)
    return torch.clamp(Phi @ w, min=0)


# =================================================================
# MAIN EXPERIMENT
# =================================================================
print("\n" + "="*60)
ok = sanity_check()
if not ok:
    print("\nStopping. Please check error messages above.")
    raise SystemExit

# Actual mesh
NX, NY = 80, 80
N_STEPS = 12
nodes, elements, d_init, n_nodes, bot, top, h = create_mesh(NX, NY)
ell = max(0.015, 2.5*h)
print(f"\n{'='*60}")
print(f"MAIN RUN: {NX}x{NY}, ell={ell:.4f}, h={h:.4f}")
print(f"{'='*60}")

u_max = 8e-3
u_steps = torch.linspace(1e-3, u_max, N_STEPS, device=device, dtype=dtype)

# Schemes
configs = {
    'A_exact':      {'type':'exact'},
    'B_grid_8':     {'type':'grid', 'n':8},
    'B_grid_16':    {'type':'grid', 'n':16},
    'B_grid_32':    {'type':'grid', 'n':32},
    'C_rbf_100':    {'type':'rbf',  'n':100},
    'C_rbf_400':    {'type':'rbf',  'n':400},
    'D_random_500': {'type':'random','n':500},
}

states = {k: {'d': d_init.clone(), 'H': torch.zeros(n_nodes, device=device, dtype=dtype)}
          for k in configs}

errors = {k: {'L2_d':[], 'Linf_d':[], 'L2_H':[], 'Linf_H':[], 'max_d':[], 'tip_err':[]}
          for k in configs}

tip = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
tip_mask = (nodes - tip).norm(dim=1) < 3*ell
print(f"Tip region: {tip_mask.sum().item()} nodes")

t0_total = time.time()
for step_i in range(N_STEPS):
    ub = u_steps[step_i].item()
    t0 = time.time()
    debug_this = (step_i == 0)  # debug first step only

    for name, cfg in configs.items():
        s = states[name]
        u_sol, d_sol, H_sol = staggered_step(
            nodes, elements, ub, s['d'], s['H'], n_nodes, bot, top, ell,
            debug=(debug_this and name == 'A_exact'))

        # Transfer H
        if cfg['type'] == 'exact':
            H_t = H_sol.clone()
        elif cfg['type'] == 'grid':
            H_t = transfer_coarse_grid(H_sol, NX, NY, cfg['n'])
        elif cfg['type'] == 'rbf':
            H_t = transfer_rbf(H_sol, nodes, cfg['n'])
        elif cfg['type'] == 'random':
            H_t = transfer_random(H_sol, nodes, NX, NY, cfg['n'], 42+step_i)

        H_t = torch.maximum(H_t, s['H'])
        s['d'] = d_sol
        s['H'] = H_t

    # Errors
    d_ex = states['A_exact']['d']
    H_ex = states['A_exact']['H']
    for name in configs:
        ds = states[name]['d']; Hs = states[name]['H']
        errors[name]['L2_d'].append((ds-d_ex).pow(2).mean().sqrt().item())
        errors[name]['Linf_d'].append((ds-d_ex).abs().max().item())
        errors[name]['L2_H'].append((Hs-H_ex).pow(2).mean().sqrt().item())
        errors[name]['Linf_H'].append((Hs-H_ex).abs().max().item())
        errors[name]['max_d'].append(ds.max().item())
        errors[name]['tip_err'].append(
            (Hs[tip_mask]-H_ex[tip_mask]).pow(2).mean().sqrt().item() if tip_mask.sum()>0 else 0)

    dt = time.time()-t0
    print(f"\nStep {step_i+1:>2}/{N_STEPS}: u={ub:.5f}, "
          f"max_d={d_ex.max():.5f}, max_H={H_ex.max():.4e}, t={dt:.1f}s")
    for name in configs:
        if name == 'A_exact': continue
        print(f"  {name:18s} L2(H)={errors[name]['L2_H'][-1]:.2e} "
              f"tip={errors[name]['tip_err'][-1]:.2e}")

print(f"\nTotal: {time.time()-t0_total:.1f}s")

# =================================================================
# PLOTS
# =================================================================
nodes_np = nodes.cpu().numpy()
tris = np.concatenate([elements[:,[0,1,2]].cpu().numpy(), elements[:,[0,2,3]].cpu().numpy()])
triang = mtri.Triangulation(nodes_np[:,0], nodes_np[:,1], tris)

# Fig 1: Fields
fig, axes = plt.subplots(1,2, figsize=(14,5))
d_np = states['A_exact']['d'].cpu().numpy()
H_np = states['A_exact']['H'].cpu().numpy()
tc = axes[0].tripcolor(triang, d_np, cmap='hot', shading='gouraud')
plt.colorbar(tc, ax=axes[0]); axes[0].set_title('d (Exact)'); axes[0].set_aspect('equal')
H_plot = np.where(H_np > 0, np.log10(H_np), -20)
tc = axes[1].tripcolor(triang, H_plot, cmap='viridis', shading='gouraud')
plt.colorbar(tc, ax=axes[1]); axes[1].set_title('log₁₀(H) (Exact)'); axes[1].set_aspect('equal')
plt.tight_layout(); plt.savefig('fig1_fields.png', dpi=150); plt.show()

# Fig 2: |ΔH| spatial
plot_names = [k for k in configs if k != 'A_exact']
n_p = len(plot_names)
ncols = min(3, n_p); nrows = (n_p+ncols-1)//ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
if n_p == 1: axes = [axes]
else: axes = axes.ravel()
for i, name in enumerate(plot_names):
    dH = (states[name]['H'] - states['A_exact']['H']).abs().cpu().numpy()
    tc = axes[i].tripcolor(triang, dH, cmap='Reds', shading='gouraud')
    plt.colorbar(tc, ax=axes[i]); axes[i].set_title(name, fontsize=10)
    axes[i].set_aspect('equal'); axes[i].plot(0.5,0.5,'g*',ms=10)
for j in range(n_p, len(axes)): axes[j].set_visible(False)
plt.suptitle('|H - H_exact| (final step)')
plt.tight_layout(); plt.savefig('fig2_H_error.png', dpi=150); plt.show()

# Fig 3: Error accumulation
fig, axes = plt.subplots(2,2, figsize=(12,10))
steps_arr = np.arange(1, N_STEPS+1)
for name, data in errors.items():
    if name == 'A_exact': continue
    mk = 'o-' if 'grid' in name else ('s--' if 'rbf' in name else '^:')
    for ax_i, key in enumerate(['L2_H','L2_d','tip_err','Linf_H']):
        ax = axes.ravel()[ax_i]
        vals = data[key]
        if any(v > 0 for v in vals):
            ax.semilogy(steps_arr, [max(v, 1e-20) for v in vals], mk, label=name, ms=4)
titles = ['L2(H)','L2(d)','Tip L2(H)','L∞(H)']
for ax, t in zip(axes.ravel(), titles):
    ax.set_title(t); ax.set_xlabel('Step'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle('Error Accumulation')
plt.tight_layout(); plt.savefig('fig3_error_accum.png', dpi=150); plt.show()

# Fig 4: H along crack line
cl = np.abs(nodes_np[:,1]-0.5) < h*0.6
x_cl = nodes_np[cl,0]; si = np.argsort(x_cl); x_cl = x_cl[si]
fig, axes = plt.subplots(1,2, figsize=(14,5))
H_ex_cl = H_np[cl][si]
axes[0].plot(x_cl, H_ex_cl, 'k-', lw=2, label='Exact')
for name in plot_names:
    Hcl = states[name]['H'].cpu().numpy()[cl][si]
    axes[0].plot(x_cl, Hcl, '--', alpha=0.7, label=name)
    axes[1].plot(x_cl, np.abs(Hcl - H_ex_cl), '-', alpha=0.7, label=name)
axes[0].set_title('H along y=0.5'); axes[0].legend(fontsize=7)
axes[1].set_title('|ΔH| along y=0.5'); axes[1].legend(fontsize=7)
for ax in axes: ax.axvline(0.5,c='gray',ls=':'); ax.set_xlabel('x'); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('fig4_crackline.png', dpi=150); plt.show()

# Fig 5: Convergence
fig, ax = plt.subplots(figsize=(8,5))
gn = ['B_grid_8','B_grid_16','B_grid_32']
ns = [8,16,32]
fH = [errors[n]['L2_H'][-1] for n in gn]
ft = [errors[n]['tip_err'][-1] for n in gn]
ax.loglog(ns, [max(v,1e-20) for v in fH], 'bo-', ms=8, label='L2(H) global')
ax.loglog(ns, [max(v,1e-20) for v in ft], 'rs-', ms=8, label='L2(H) tip')
ax.set_xlabel('N storage'); ax.set_ylabel('Error'); ax.set_title('Convergence')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('fig5_convergence.png', dpi=150); plt.show()

# =================================================================
# VERDICT
# =================================================================
print("\n" + "="*60)
print("VERDICT")
print("="*60)
print(f"\n{'Scheme':<18} | {'L2(H)':>10} | {'Tip err':>10} | {'Growth':>8}")
print("-"*55)
for name in configs:
    if name == 'A_exact': continue
    fH = errors[name]['L2_H'][-1]
    ft = errors[name]['tip_err'][-1]
    nz = [e for e in errors[name]['L2_H'] if e > 1e-15]
    gr = f"{nz[-1]/nz[0]:.1f}x" if len(nz)>=2 and nz[0]>1e-15 else "N/A"
    print(f"{name:<18} | {fH:>10.2e} | {ft:>10.2e} | {gr:>8}")

sig = any(errors[n]['tip_err'][-1] > 1e-8 for n in configs if n != 'A_exact')
print(f"\n{'✅ T1 VALIDATED' if sig else '⚠️ Need more loading or finer analysis'}")
print("="*60)

with open('t1_2d_results.json','w') as f:
    json.dump({'errors':{k:{kk:vv for kk,vv in v.items()} for k,v in errors.items()},
               'params':{'E':E,'nu':nu,'Gc':Gc,'ell':ell,'NX':NX,'NY':NY}}, f, indent=2)
print("Results saved.")
