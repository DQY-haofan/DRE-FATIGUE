"""
T2 Verification: Miehe vs Bourdin Irreversibility under Mesh-Free Transfer
==========================================================================
Copy into Colab cell, run with GPU. ~3-5 min on T4.

KEY QUESTION: Does Bourdin's approach (d >= d_prev, no H variable)
avoid the history transfer catastrophe shown in T1?

Two irreversibility approaches:
  MIEHE:   H(x) = max_t psi_e+(x,t), d-equation driven by H
  BOURDIN: d-equation driven by psi_e+(current), then clamp d >= d_prev

For each approach, test exact vs coarse transfer of the carried variable:
  MIEHE carries H  -> coarse H transfer
  BOURDIN carries d_prev -> coarse d_prev transfer

Expected: d is smoother than H (regularized by ell) -> less transfer error.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
print(f"Device: {device}")

# ========== Physics ==========
E, nu = 210.0, 0.3
Gc = 2.7e-3
k_res = 1e-7
lam = E*nu/((1+nu)*(1-2*nu))
mu  = E/(2*(1+nu))

# ========== Gauss quadrature ==========
gp = 1.0/3**0.5
GP_XI  = torch.tensor([-gp, gp, gp, -gp], device=device, dtype=dtype)
GP_ETA = torch.tensor([-gp, -gp, gp, gp], device=device, dtype=dtype)
GP_W   = torch.ones(4, device=device, dtype=dtype)

def shape_at_gp(xi, eta):
    N = 0.25*torch.stack([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])
    dN = 0.25*torch.stack([
        torch.stack([-(1-eta),(1-eta),(1+eta),-(1+eta)]),
        torch.stack([-(1-xi),-(1+xi),(1+xi),(1-xi)])])
    return N, dN

N_gps, dN_gps = [], []
for q in range(4):
    N_q, dN_q = shape_at_gp(GP_XI[q], GP_ETA[q])
    N_gps.append(N_q); dN_gps.append(dN_q)

# ========== Mesh ==========
NX, NY = 80, 80
h = 1.0/max(NX, NY)
ell = max(0.015, 2.5*h)

def create_mesh():
    x = torch.linspace(0,1,NX+1,device=device,dtype=dtype)
    y = torch.linspace(0,1,NY+1,device=device,dtype=dtype)
    Y,X = torch.meshgrid(y,x,indexing='ij')
    nodes = torch.stack([X.reshape(-1),Y.reshape(-1)],dim=1)
    n_nodes = nodes.shape[0]
    j = torch.arange(NY,device=device).unsqueeze(1).expand(NY,NX).reshape(-1)
    i = torch.arange(NX,device=device).unsqueeze(0).expand(NY,NX).reshape(-1)
    n0 = j*(NX+1)+i
    elements = torch.stack([n0,n0+1,n0+(NX+1)+1,n0+(NX+1)],dim=1).long()
    crack = (torch.abs(nodes[:,1]-0.5)<h*0.6) & (nodes[:,0]<=0.5+h*0.6)
    d_init = torch.zeros(n_nodes,device=device,dtype=dtype)
    d_init[crack] = 1.0
    bot = nodes[:,1]<1e-10; top = nodes[:,1]>1.0-1e-10
    return nodes, elements, d_init, n_nodes, bot, top

# ========== FEM utilities ==========
def get_dN_dx(dN_q, xe):
    J = torch.einsum('ij,ejk->eik', dN_q, xe)
    detJ = J[:,0,0]*J[:,1,1]-J[:,0,1]*J[:,1,0]
    inv = 1.0/detJ
    invJ = torch.zeros_like(J)
    invJ[:,0,0]=J[:,1,1]*inv; invJ[:,0,1]=-J[:,0,1]*inv
    invJ[:,1,0]=-J[:,1,0]*inv; invJ[:,1,1]=J[:,0,0]*inv
    return torch.einsum('eij,jk->eik', invJ, dN_q), detJ

def build_B(dN_dx, ne):
    B = torch.zeros(ne,3,8,device=device,dtype=dtype)
    for i in range(4):
        B[:,0,2*i]=dN_dx[:,0,i]; B[:,1,2*i+1]=dN_dx[:,1,i]
        B[:,2,2*i]=dN_dx[:,1,i]; B[:,2,2*i+1]=dN_dx[:,0,i]
    return B

def elem_dofs(el):
    return torch.stack([2*el,2*el+1],dim=-1).reshape(el.shape[0],8)

# ========== Solve u ==========
def solve_u(nodes, elements, d_field, bot, top, u_bar, n_nodes):
    ne = elements.shape[0]; ndof = 2*n_nodes
    xe = nodes[elements]; de = d_field[elements]; dofs = elem_dofs(elements)
    D0 = torch.tensor([[lam+2*mu,lam,0],[lam,lam+2*mu,0],[0,0,mu]],device=device,dtype=dtype)
    Ke = torch.zeros(ne,8,8,device=device,dtype=dtype)
    for q in range(4):
        dN_dx, detJ = get_dN_dx(dN_gps[q], xe)
        B = build_B(dN_dx, ne)
        d_gp = (N_gps[q].unsqueeze(0)*de).sum(1)
        g = (1-d_gp)**2+k_res
        BtDB = torch.einsum('eji,jk,ekl->eil',B,D0,B)
        Ke += (g*detJ*GP_W[q]).unsqueeze(-1).unsqueeze(-1)*BtDB
    row = dofs.unsqueeze(2).expand(-1,8,8).reshape(-1)
    col = dofs.unsqueeze(1).expand(-1,8,8).reshape(-1)
    K = torch.zeros(ndof,ndof,device=device,dtype=dtype)
    K.index_put_((row,col), Ke.reshape(-1), accumulate=True)
    f = torch.zeros(ndof,device=device,dtype=dtype)
    pen = K.diag().max()*1e7
    for dof_ids, val in [(2*torch.where(bot)[0],0.),(2*torch.where(bot)[0]+1,0.),
                          (2*torch.where(top)[0]+1,u_bar)]:
        K[dof_ids,:]=0; K[dof_ids,dof_ids]=pen; f[dof_ids]=pen*val
    return torch.linalg.solve(K,f)

# ========== Compute psi_e+ at nodes ==========
def compute_psi_plus(nodes, elements, u_field, n_nodes):
    """Returns psi_e+ at nodes (current step only, no max)."""
    ne = elements.shape[0]; xe = nodes[elements]
    dofs = elem_dofs(elements); ue = u_field[dofs]
    psi_acc = torch.zeros(n_nodes,device=device,dtype=dtype)
    wt = torch.zeros(n_nodes,device=device,dtype=dtype)
    for q in range(4):
        dN_dx, detJ = get_dN_dx(dN_gps[q], xe)
        B = build_B(dN_dx, ne)
        eps = torch.einsum('eij,ej->ei', B, ue)
        exx,eyy,gxy = eps[:,0],eps[:,1],eps[:,2]*0.5
        tr = exx+eyy; det = exx*eyy-gxy**2
        disc = torch.clamp(tr**2-4*det,min=0)
        e1 = 0.5*(tr+torch.sqrt(disc)); e2 = 0.5*(tr-torch.sqrt(disc))
        tr_p = torch.clamp(tr,min=0)
        psi = 0.5*lam*tr_p**2 + mu*(torch.clamp(e1,min=0)**2+torch.clamp(e2,min=0)**2)
        for i in range(4):
            w_i = N_gps[q][i].abs()
            psi_acc.scatter_add_(0, elements[:,i], psi*w_i*detJ)
            wt.scatter_add_(0, elements[:,i], w_i*detJ*torch.ones(ne,device=device,dtype=dtype))
    return psi_acc/wt.clamp(min=1e-30)

# ========== Solve d ==========
def solve_d_general(nodes, elements, driving_field, d_lower_bound, n_nodes):
    """
    Solve phase-field driven by `driving_field` (could be H or psi_e+).
    Clamp d >= d_lower_bound (could be d_prev or d_init).
    """
    ne = elements.shape[0]; xe = nodes[elements]
    drv_e = driving_field[elements]
    Ke = torch.zeros(ne,4,4,device=device,dtype=dtype)
    fe = torch.zeros(ne,4,device=device,dtype=dtype)
    for q in range(4):
        dN_dx, detJ = get_dN_dx(dN_gps[q], xe)
        drv_gp = (N_gps[q].unsqueeze(0)*drv_e).sum(1)
        N_out = torch.einsum('i,j->ij', N_gps[q], N_gps[q])
        grad = Gc*ell*torch.einsum('eji,ejk->eik', dN_dx, dN_dx)
        mass_c = Gc/(2*ell)+2*drv_gp
        mass = mass_c.unsqueeze(-1).unsqueeze(-1)*N_out.unsqueeze(0)
        w = (GP_W[q]*detJ).unsqueeze(-1).unsqueeze(-1)
        Ke += (grad+mass)*w
        fe += (2*drv_gp*GP_W[q]*detJ).unsqueeze(-1)*N_gps[q].unsqueeze(0)
    row = elements.unsqueeze(2).expand(-1,4,4).reshape(-1)
    col = elements.unsqueeze(1).expand(-1,4,4).reshape(-1)
    Kd = torch.zeros(n_nodes,n_nodes,device=device,dtype=dtype)
    Kd.index_put_((row,col), Ke.reshape(-1), accumulate=True)
    rhs = torch.zeros(n_nodes,device=device,dtype=dtype)
    rhs.scatter_add_(0, elements.reshape(-1), fe.reshape(-1))
    d = torch.linalg.solve(Kd, rhs)
    d = torch.clamp(torch.maximum(d, d_lower_bound), 0, 1)
    return d

# ========== Transfer schemes ==========
def transfer_exact(field, *args):
    return field.clone()

def transfer_coarse(field, n_storage):
    """Downsample to n_storage x n_storage, upsample back."""
    fg = field.reshape(NY+1,NX+1).unsqueeze(0).unsqueeze(0)
    coarse = F.interpolate(fg, size=(n_storage,n_storage), mode='bilinear', align_corners=True)
    recon = F.interpolate(coarse, size=(NY+1,NX+1), mode='bilinear', align_corners=True)
    return torch.clamp(recon.reshape(-1), min=0)

def transfer_rbf(field, nodes, n_centers):
    """RBF regression."""
    nn = nodes.shape[0]
    idx = torch.linspace(0,nn-1,n_centers,device=device).long()
    c = nodes[idx]; fc = field[idx]
    sig = 1.0/n_centers**0.5
    diff = nodes.unsqueeze(1)-c.unsqueeze(0)
    Phi = torch.exp(-(diff**2).sum(-1)/(2*sig**2))
    w = torch.linalg.solve(Phi[idx]+1e-6*torch.eye(n_centers,device=device,dtype=dtype), fc)
    return torch.clamp(Phi@w, min=0)

# ========== Staggered solvers ==========
def stagger_miehe(nodes, elements, u_bar, d_prev, H_prev, n_nodes, bot, top,
                  max_it=20, tol=1e-6):
    """Miehe approach: H = max(H_prev, psi_e+), d driven by H."""
    d = d_prev.clone(); H = H_prev.clone()
    for it in range(max_it):
        d_old = d.clone()
        u = solve_u(nodes, elements, d, bot, top, u_bar, n_nodes)
        psi = compute_psi_plus(nodes, elements, u, n_nodes)
        H = torch.maximum(H, psi)
        d = solve_d_general(nodes, elements, H, d_prev, n_nodes)
        if (d-d_old).abs().max()<tol: break
    return u, d, H, psi

def stagger_bourdin(nodes, elements, u_bar, d_prev, n_nodes, bot, top,
                    max_it=20, tol=1e-6):
    """Bourdin approach: d driven by psi_e+ (current), clamp d >= d_prev."""
    d = d_prev.clone()
    for it in range(max_it):
        d_old = d.clone()
        u = solve_u(nodes, elements, d, bot, top, u_bar, n_nodes)
        psi = compute_psi_plus(nodes, elements, u, n_nodes)
        d = solve_d_general(nodes, elements, psi, d_prev, n_nodes)
        if (d-d_old).abs().max()<tol: break
    return u, d, psi


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================
print(f"\nMesh: {NX}x{NY}, ell={ell:.4f}, h={h:.4f}")
nodes, elements, d_init, n_nodes, bot, top = create_mesh()

N_STEPS = 12
u_max = 8e-3
u_steps = torch.linspace(1e-3, u_max, N_STEPS, device=device, dtype=dtype)

tip = torch.tensor([0.5,0.5],device=device,dtype=dtype)
tip_mask = (nodes-tip).norm(dim=1) < 3*ell
print(f"Tip nodes: {tip_mask.sum().item()}")

# ---- Define all runs ----
# Format: (name, method, transfer_type, n_storage)
# For Miehe: transferred variable = H
# For Bourdin: transferred variable = d_prev
runs = [
    # Baselines (exact)
    ('Miehe_exact',  'miehe',  'exact', 0),
    ('Bourdin_exact','bourdin','exact', 0),
    # Miehe with coarse H transfer
    ('Miehe_H_grid8',  'miehe','grid', 8),
    ('Miehe_H_grid16', 'miehe','grid', 16),
    ('Miehe_H_grid32', 'miehe','grid', 32),
    ('Miehe_H_rbf100', 'miehe','rbf', 100),
    # Bourdin with coarse d_prev transfer
    ('Bourdin_d_grid8',  'bourdin','grid', 8),
    ('Bourdin_d_grid16', 'bourdin','grid', 16),
    ('Bourdin_d_grid32', 'bourdin','grid', 32),
    ('Bourdin_d_rbf100', 'bourdin','rbf', 100),
]

# State: each run tracks d, H (miehe only), and errors
class RunState:
    def __init__(self, d_init, n_nodes):
        self.d = d_init.clone()
        self.H = torch.zeros(n_nodes, device=device, dtype=dtype)
        self.errs = {'L2_d':[], 'Linf_d':[], 'tip_d':[], 'L2_carried':[], 'tip_carried':[]}

states = {name: RunState(d_init, n_nodes) for name,_,_,_ in runs}

t0_all = time.time()
for step_i in range(N_STEPS):
    ub = u_steps[step_i].item()
    t0 = time.time()

    for name, method, tf_type, tf_n in runs:
        s = states[name]

        if method == 'miehe':
            u, d, H, psi = stagger_miehe(nodes, elements, ub, s.d, s.H, n_nodes, bot, top)
            # Transfer H
            if tf_type == 'exact':
                H_t = H.clone()
            elif tf_type == 'grid':
                H_t = transfer_coarse(H, tf_n)
                H_t = torch.maximum(H_t, s.H)  # monotonicity
            elif tf_type == 'rbf':
                H_t = transfer_rbf(H, nodes, tf_n)
                H_t = torch.maximum(H_t, s.H)
            s.H = H_t
            s.d = d

        elif method == 'bourdin':
            u, d, psi = stagger_bourdin(nodes, elements, ub, s.d, n_nodes, bot, top)
            # Transfer d_prev
            if tf_type == 'exact':
                d_t = d.clone()
            elif tf_type == 'grid':
                d_t = transfer_coarse(d, tf_n)
                d_t = torch.maximum(d_t, s.d)  # monotonicity
            elif tf_type == 'rbf':
                d_t = transfer_rbf(d, nodes, tf_n)
                d_t = torch.clamp(torch.maximum(d_t, s.d), 0, 1)
            s.d = d_t

    # Compute errors vs respective exact baselines
    d_miehe_ex = states['Miehe_exact'].d
    H_miehe_ex = states['Miehe_exact'].H
    d_bourdin_ex = states['Bourdin_exact'].d

    for name, method, tf_type, tf_n in runs:
        s = states[name]
        if method == 'miehe':
            ref_d = d_miehe_ex; ref_carried = H_miehe_ex
            carried = s.H
        else:
            ref_d = d_bourdin_ex; ref_carried = d_bourdin_ex
            carried = s.d

        s.errs['L2_d'].append((s.d - ref_d).pow(2).mean().sqrt().item())
        s.errs['Linf_d'].append((s.d - ref_d).abs().max().item())
        s.errs['tip_d'].append(
            (s.d[tip_mask]-ref_d[tip_mask]).pow(2).mean().sqrt().item() if tip_mask.sum()>0 else 0)
        s.errs['L2_carried'].append((carried-ref_carried).pow(2).mean().sqrt().item())
        s.errs['tip_carried'].append(
            (carried[tip_mask]-ref_carried[tip_mask]).pow(2).mean().sqrt().item() if tip_mask.sum()>0 else 0)

    dt = time.time()-t0
    # Print summary
    print(f"\nStep {step_i+1:>2}/{N_STEPS}: u={ub:.5f}, t={dt:.1f}s")
    print(f"  Miehe_exact:  max_d={d_miehe_ex.max():.5f}  max_H={H_miehe_ex.max():.4e}")
    print(f"  Bourdin_exact: max_d={d_bourdin_ex.max():.5f}")
    for name, method, tf_type, tf_n in runs:
        if tf_type == 'exact': continue
        s = states[name]
        tag = 'H' if method=='miehe' else 'd'
        print(f"  {name:22s} L2({tag})={s.errs['L2_carried'][-1]:.2e}  "
              f"tip({tag})={s.errs['tip_carried'][-1]:.2e}  "
              f"L2(d)={s.errs['L2_d'][-1]:.2e}")

total_t = time.time()-t0_all
print(f"\nTotal: {total_t:.1f}s")


# ==========================================================
# PLOTS
# ==========================================================
nodes_np = nodes.cpu().numpy()
tris = np.concatenate([elements[:,[0,1,2]].cpu().numpy(), elements[:,[0,2,3]].cpu().numpy()])
triang = mtri.Triangulation(nodes_np[:,0], nodes_np[:,1], tris)
steps_arr = np.arange(1, N_STEPS+1)

# ----- Fig 1: Baselines comparison -----
fig, axes = plt.subplots(1,3, figsize=(18,5))
for ax, (name, label) in zip(axes[:2], [('Miehe_exact','Miehe d'),('Bourdin_exact','Bourdin d')]):
    d_np = states[name].d.cpu().numpy()
    tc = ax.tripcolor(triang, d_np, cmap='hot', shading='gouraud', vmin=0, vmax=1)
    plt.colorbar(tc,ax=ax); ax.set_title(label); ax.set_aspect('equal')
# Difference
dd = (states['Miehe_exact'].d - states['Bourdin_exact'].d).abs().cpu().numpy()
tc = axes[2].tripcolor(triang, dd, cmap='Blues', shading='gouraud')
plt.colorbar(tc,ax=axes[2]); axes[2].set_title('|d_Miehe - d_Bourdin|'); axes[2].set_aspect('equal')
plt.suptitle('Baseline Comparison: Miehe vs Bourdin (exact storage)')
plt.tight_layout(); plt.savefig('t2_fig1_baselines.png',dpi=150); plt.show()

# ----- Fig 2: THE KEY PLOT - Error comparison -----
fig, axes = plt.subplots(2,2, figsize=(14,10))

# Top-left: L2 error of carried variable
ax = axes[0,0]; ax.set_title('L2 error of CARRIED variable\n(H for Miehe, d for Bourdin)')
for name, method, tf_type, tf_n in runs:
    if tf_type == 'exact': continue
    s = states[name]
    vals = s.errs['L2_carried']
    if any(v>0 for v in vals):
        mk = 'o-' if method=='miehe' else 's--'
        c = 'C0' if 'grid8' in name else ('C1' if 'grid16' in name else ('C2' if 'grid32' in name else 'C3'))
        ax.semilogy(steps_arr, [max(v,1e-20) for v in vals], mk, label=name, ms=4, color=c if method=='miehe' else None)
ax.legend(fontsize=7); ax.grid(True,alpha=0.3); ax.set_xlabel('Step')

# Top-right: Tip error of carried variable
ax = axes[0,1]; ax.set_title('Crack tip error of CARRIED variable')
for name, method, tf_type, tf_n in runs:
    if tf_type == 'exact': continue
    s = states[name]
    vals = s.errs['tip_carried']
    if any(v>0 for v in vals):
        mk = 'o-' if method=='miehe' else 's--'
        ax.semilogy(steps_arr, [max(v,1e-20) for v in vals], mk, label=name, ms=4)
ax.legend(fontsize=7); ax.grid(True,alpha=0.3); ax.set_xlabel('Step')

# Bottom-left: L2(d) error
ax = axes[1,0]; ax.set_title('L2(d) error vs respective exact baseline')
for name, method, tf_type, tf_n in runs:
    if tf_type == 'exact': continue
    s = states[name]
    vals = s.errs['L2_d']
    if any(v>0 for v in vals):
        mk = 'o-' if method=='miehe' else 's--'
        ax.semilogy(steps_arr, [max(v,1e-20) for v in vals], mk, label=name, ms=4)
ax.legend(fontsize=7); ax.grid(True,alpha=0.3); ax.set_xlabel('Step')

# Bottom-right: Tip d error
ax = axes[1,1]; ax.set_title('Crack tip L2(d) error')
for name, method, tf_type, tf_n in runs:
    if tf_type == 'exact': continue
    s = states[name]
    vals = s.errs['tip_d']
    if any(v>0 for v in vals):
        mk = 'o-' if method=='miehe' else 's--'
        ax.semilogy(steps_arr, [max(v,1e-20) for v in vals], mk, label=name, ms=4)
ax.legend(fontsize=7); ax.grid(True,alpha=0.3); ax.set_xlabel('Step')

plt.suptitle('T2: Miehe (—○—) vs Bourdin (--□--) under coarse transfer', fontsize=14)
plt.tight_layout(); plt.savefig('t2_fig2_comparison.png',dpi=150); plt.show()

# ----- Fig 3: Spatial error maps -----
fig, axes = plt.subplots(2,4, figsize=(20,9))
# Row 1: Miehe |ΔH|
miehe_schemes = [(n,m,t,s) for n,m,t,s in runs if m=='miehe' and t!='exact']
for i,(name,_,_,_) in enumerate(miehe_schemes):
    ax = axes[0,i]
    dH = (states[name].H - states['Miehe_exact'].H).abs().cpu().numpy()
    tc = ax.tripcolor(triang, dH, cmap='Reds', shading='gouraud')
    plt.colorbar(tc,ax=ax); ax.set_title(f'{name}\n|ΔH|',fontsize=9)
    ax.set_aspect('equal'); ax.plot(0.5,0.5,'g*',ms=10)
# Row 2: Bourdin |Δd|
bourdin_schemes = [(n,m,t,s) for n,m,t,s in runs if m=='bourdin' and t!='exact']
for i,(name,_,_,_) in enumerate(bourdin_schemes):
    ax = axes[1,i]
    dd = (states[name].d - states['Bourdin_exact'].d).abs().cpu().numpy()
    tc = ax.tripcolor(triang, dd, cmap='Reds', shading='gouraud')
    plt.colorbar(tc,ax=ax); ax.set_title(f'{name}\n|Δd|',fontsize=9)
    ax.set_aspect('equal'); ax.plot(0.5,0.5,'g*',ms=10)
plt.suptitle('Spatial error: Miehe |ΔH| (top) vs Bourdin |Δd| (bottom)', fontsize=13)
plt.tight_layout(); plt.savefig('t2_fig3_spatial.png',dpi=150); plt.show()

# ----- Fig 4: d along crack line -----
fig, axes = plt.subplots(1,2, figsize=(14,5))
cl = np.abs(nodes_np[:,1]-0.5)<h*0.6
x_cl = nodes_np[cl,0]; si = np.argsort(x_cl); x_cl = x_cl[si]

for ax, (exact_name, schemes, title) in zip(axes, [
    ('Miehe_exact', miehe_schemes, 'Miehe: d along y=0.5'),
    ('Bourdin_exact', bourdin_schemes, 'Bourdin: d along y=0.5')]):
    d_ex = states[exact_name].d.cpu().numpy()[cl][si]
    ax.plot(x_cl, d_ex, 'k-', lw=2, label='Exact')
    for name,_,_,_ in schemes:
        d_s = states[name].d.cpu().numpy()[cl][si]
        ax.plot(x_cl, d_s, '--', alpha=0.7, label=name)
    ax.set_title(title); ax.legend(fontsize=7); ax.axvline(0.5,c='gray',ls=':')
    ax.set_xlabel('x'); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('t2_fig4_crackline.png',dpi=150); plt.show()


# ==========================================================
# VERDICT
# ==========================================================
print("\n" + "="*70)
print("T2 VERDICT: Miehe vs Bourdin under mesh-free transfer")
print("="*70)

print(f"\n{'Run':<24} | {'L2(carried)':>11} | {'Tip(carried)':>12} | {'L2(d)':>10} | {'Tip(d)':>10}")
print("-"*78)
for name, method, tf_type, tf_n in runs:
    if tf_type == 'exact': continue
    s = states[name]
    tag = 'H' if method=='miehe' else 'd'
    print(f"{name:<24} | {s.errs['L2_carried'][-1]:>11.2e} | "
          f"{s.errs['tip_carried'][-1]:>12.2e} | "
          f"{s.errs['L2_d'][-1]:>10.2e} | {s.errs['tip_d'][-1]:>10.2e}")

# Compare matched pairs
print("\n--- Matched pair comparison (grid16) ---")
m_name = 'Miehe_H_grid16'; b_name = 'Bourdin_d_grid16'
m_tip = states[m_name].errs['tip_carried'][-1]
b_tip = states[b_name].errs['tip_carried'][-1]
ratio = m_tip / b_tip if b_tip > 1e-20 else float('inf')
print(f"  Miehe  tip error: {m_tip:.4e}")
print(f"  Bourdin tip error: {b_tip:.4e}")
print(f"  Ratio (Miehe/Bourdin): {ratio:.1f}x")

m_d = states[m_name].errs['L2_d'][-1]
b_d = states[b_name].errs['L2_d'][-1]
ratio_d = m_d / b_d if b_d > 1e-20 else float('inf')
print(f"  Miehe  L2(d): {m_d:.4e}")
print(f"  Bourdin L2(d): {b_d:.4e}")
print(f"  Ratio: {ratio_d:.1f}x")

if ratio > 5:
    print(f"""
✅ T2 CONFIRMED: Bourdin's approach is {ratio:.0f}x more robust to coarse transfer.
   H has sharp crack-tip gradients → large transfer error.
   d is ℓ-regularized (smooth) → much smaller transfer error.
   
   IMPLICATION FOR PAPER: In mesh-free DEM, Bourdin's irreversibility
   (d >= d_prev) is strongly preferred over Miehe's history variable H.
   This is a novel finding - connects variational formulation choice
   to mesh-free implementability.""")
elif ratio > 1.5:
    print(f"\n⚠️ Bourdin is {ratio:.1f}x better but difference is moderate.")
else:
    print(f"\n🔍 Comparable performance (ratio={ratio:.1f}x). Need further analysis.")

print("\nFigures saved: t2_fig1-4.")
