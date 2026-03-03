# Paper Outline — ML4CSE Submission

## Working Title

**On the Structural Instability of History Variable Transfer in Mesh-Free Neural PDE Solvers for Path-Dependent Materials**

*Alternative (shorter):*
**History Variable Transfer Error in Neural Solvers for Dissipative Mechanics: Analysis and Remedies**

---

## Target

- **Journal**: Machine Learning for Computational Science and Engineering (Springer #44379)
- **Topical Collection**: "Accelerating Numerical Methods With Scientific Machine Learning" (Goswami & Kumar)
- **Format**: Original research article, no page limit, Springer LaTeX template, numbered references [1]

---

## Abstract (~200 words)

Neural network-based PDE solvers (Deep Ritz, PINNs) for mechanics typically approximate displacement or phase fields as continuous NN functions, eliminating mesh dependence. However, for path-dependent materials (plasticity, damage, fatigue), internal state variables from previous load steps must be stored and transferred—a process that reintroduces discrete approximation. We identify this **history variable transfer** as a fundamental, previously unanalyzed source of error in mesh-free solvers. Through rigorous analysis and numerical verification on phase-field fracture benchmarks:

1. We prove that naive mesh-free transfer of the Miehe history variable H = max ψ_e⁺ leads to exponential error accumulation (60,000× growth over 12 load steps), concentrated at crack tips.
2. We show that Bourdin's irreversibility constraint (d ≥ d_prev) is 77× more robust under coarse transfer, due to three mechanisms: range boundedness, ℓ-regularization, and absence of max-operator ratcheting.
3. We extend the analysis to fatigue (Carrara model), proving it admits an incremental variational reformulation and identifying that fatigue triples the history transfer burden with two additional unbounded fields.
4. We propose a **bounded transfer principle**: mesh-free solvers should carry only bounded variables, using variational reformulation or contractive maps to eliminate unbounded history fields.

**Keywords**: Deep Ritz Method, phase-field fracture, history variables, error accumulation, mesh-free methods, fatigue, incremental variational principles

---

## 1. Introduction (2–3 pages)

### 1.1 Context: Neural PDE Solvers for Mechanics
- Deep Ritz Method (E & Yu, 2018), PINNs (Raissi et al., 2019) → mesh-free approximation of displacement fields
- Success in linear elasticity, Stokes flow, etc. → growing interest in nonlinear/dissipative problems
- Phase-field fracture as variational model (Francfort-Marigo, 1998; Bourdin et al., 2000) → natural fit for energy-based NN solvers

### 1.2 The History Variable Problem
- Path-dependent materials: plasticity (ε^p, α), damage (d, H), fatigue (ᾱ) require storing internal state variables across load steps
- Classical FEM: stored at Gauss quadrature points on a fixed mesh → well-controlled
- Mesh-free NN solvers: NN outputs continuous fields, but history from step n must be sampled, stored, and fed into step n+1
- **This transfer step reintroduces discrete approximation in an otherwise continuous framework**
- Surprisingly, NO prior work analyzes this transfer error or its accumulation

### 1.3 Contributions
Enumerate 4 contributions matching the Abstract. State clearly: "To our knowledge, this is the first systematic analysis of history variable transfer error in neural PDE solvers for dissipative mechanics."

### 1.4 Paper Organization
Brief roadmap of Sections 2–6.

---

## 2. Problem Formulation (3–4 pages)

### 2.1 Phase-Field Fracture: Variational Framework
- Energy functional E[u, d] = E_bulk + E_frac
- AT2 model, spectral split (Miehe et al., 2010)
- Staggered solution: alternate u-solve and d-solve

### 2.2 Two Irreversibility Approaches
- **Miehe (2010)**: H(x,t) = max_{s≤t} ψ_e⁺(x,s), d driven by H
  - H stored at quadrature points, updated each step
- **Bourdin (2000) / Gerasimov-De Lorenzis (2019)**: d ≥ d_prev, d driven by current ψ_e⁺
  - Variationally consistent (Gerasimov & De Lorenzis, CMAME 2019)
  - d_prev stored and transferred

### 2.3 Neural Network Approximation
- Deep Ritz: u_θ(x), d_φ(x) as NNs, minimize energy w.r.t. θ, φ
- Cite: Manav et al. (2024) — DRM for phase-field fracture
- At each load step: retrain NN → obtain continuous u, d fields
- **The transfer step**: from continuous d(x) at step n → discrete storage → reconstruction at step n+1

### 2.4 History Variable Transfer as Approximation Operator
- **Definition**: Transfer operator T_N: L²(Ω) → L²(Ω) parameterized by storage resolution N
  - T_grid_N: bilinear interpolation on N×N grid
  - T_rbf_N: RBF regression with N centers  
  - T_nn_N: NN regression with N parameters
- **Single-step error**: e_n = ||T_N[f_n] - f_n||
- **Accumulated error**: E_n = ||f̃_n - f_n^{exact}|| where f̃ uses transferred history
- Key question: How does E_n grow with n?

---

## 3. Analysis of History Transfer Error (4–5 pages) — CORE CONTRIBUTION

### 3.1 Proposition 1 (T1): Exponential Error Accumulation
**Statement**: Under Miehe's scheme with transfer operator T_N, the L² error in H satisfies:

  E_n(H) ≥ E_1(H) · ∏_{k=1}^{n-1} (1 + γ_k)

where γ_k > 0 depends on the nonlinear coupling H → d → u → ψ_e⁺.

**Proof sketch**:
- Step n: H̃_n = max(T_N[H̃_{n-1}], ψ̃_e⁺_n)
- Transfer error: T_N[H_{n-1}] = H_{n-1} + δ_n (where δ_n is transfer noise)
- Max operator: preserves overestimates in δ_n (one-sided ratchet)
- Coupling: ψ̃_e⁺_n = ψ_e⁺(ε(u(d(H̃_{n-1})))) ≠ ψ_e⁺_n^{exact}
- Result: errors compound multiplicatively

**Numerical verification** (Section 5.1):
- SEN-T benchmark, 80×80 mesh, 12 load steps
- 6 transfer schemes: grid (8,16,32), RBF (100,400), random (500)
- L2(H) growth: 7.5e-5 → 4.49 = **60,000× in 12 steps**
- Error spatially concentrated at crack tip (Figs. 2–4)

### 3.2 Proposition 2 (T2): Bourdin's Structural Advantage
**Statement**: Under the same transfer operator T_N, Bourdin's d_prev transfer yields bounded error:

  E_n(d) ≤ C · h_s/ℓ

where h_s = 1/N is the storage spacing and ℓ is the phase-field length scale.

**Three mechanisms**:
1. **Range boundedness**: d ∈ [0,1] → |δ_n| ≤ 1 always (vs. H ∈ [0,∞))
2. **ℓ-regularization**: ||∇d||_∞ ≤ C/ℓ bounded (vs. ||∇H|| → ∞ at crack tip in 2D)
3. **No max-ratcheting**: d ≥ d_prev with d ∈ [0,1] → overestimates self-limit via saturation

**Numerical verification** (Section 5.2):
- Same SEN-T, Miehe vs Bourdin with matched transfer grids
- Grid-16: Miehe tip error 11.88 vs Bourdin tip error 0.155 → **77× ratio**
- Bourdin error converges: grid 8→16→32 gives 0.381→0.155→0.051 ≈ O(1/N)
- Miehe error does NOT converge: grid32 worse than grid16 in L2(d)

**1D analytical support**:
- Coupled 1D staggered solve confirms ratio ~160–187×
- Identifies Mechanism A (2D crack-tip singularity) as amplifier in 2D

### 3.3 The Bounded Transfer Principle
**Definition**: A history variable f is *transfer-compatible* if:
1. f has bounded range: f ∈ [a, b]
2. f has bounded gradient: ||∇f||_∞ ≤ C < ∞
3. The update operator for f does not create one-sided drift

**Theorem (informal)**: For transfer-compatible variables, the accumulated error E_n is bounded uniformly in n (for fixed transfer resolution N).

**Corollary**: In mesh-free DEM, one should reformulate the variational principle to carry ONLY transfer-compatible variables. Specifically:
- Phase-field fracture: use Bourdin (carry d) instead of Miehe (carry H)
- Plasticity: carry g(ε^p) or normalized quantities rather than raw plastic strain

---

## 4. Extension to Fatigue (3–4 pages)

### 4.1 Carrara Fatigue Model Review
- Gc → f(ᾱ)·Gc where ᾱ = ∫⟨ψ̇_e⁺⟩₊ dt (accumulated fatigue variable)
- f(ᾱ) = (1 + 2p(1-ξ)ᾱ)^{-1/(1-ξ)} (asymptotic degradation)
- Cite: Carrara, Ambati, Alessi, De Lorenzis (2020)

### 4.2 Proposition 3 (T3): Incremental Variational Reformulation
**Theorem**: Carrara's fatigue phase-field admits a condensed incremental potential:

  Π̃_{n+1}[u,d] = E_bulk(u,d) + f(ᾱ_{n+1}(u)) · Gc · ∫γ(d,∇d) dΩ

where ᾱ_{n+1}(u) = ᾱ_n + ⟨ψ_e⁺(ε(u)) - ψ_e⁺(ε_n)⟩₊

**Proof**: Direct verification that δ_d Π̃ = 0 recovers Carrara's Eq. (21).

**Remark**: δ_u Π̃ = 0 reveals additional coupling term f'·Gc·γ·∂ψ_e⁺/∂ε absent in Carrara's staggered scheme.

### 4.3 Fatigue Amplifies the Transfer Problem
- Standard phase-field: 1 transferred field (d_prev, bounded)
- Fatigue: 3 fields (d_prev ∈ [0,1], ᾱ ∈ [0,∞), ψ_e⁺_prev ∈ [0,∞))
- Two of three are UNBOUNDED → same pathology as Miehe's H
- Cyclic loading: hundreds/thousands of cycles → hundreds/thousands of transfers
- Numerical: ᾱ error grows 1180× over 20 cycles (200 transfers) with N_s=16

### 4.4 Compressed Fatigue Transfer (Constructive Contribution)
- Key observation: f: [0,∞) → [0,1] is a **contractive map**
- Transfer f(ᾱ) instead of ᾱ:
  - f(ᾱ) ∈ [0,1] → bounded, transfer-compatible
  - err(f) = |f'| · err(ᾱ) ≤ err(ᾱ)
  - Same mechanism as Bourdin's advantage over Miehe
- Trade-off: f is not invertible → cannot recover ᾱ from f(ᾱ)
  - But ᾱ is only needed THROUGH f in the phase-field equation
  - So transfer of f(ᾱ) is **sufficient** for the d-equation

---

## 5. Numerical Experiments (4–5 pages)

### 5.1 Experiment 1: Error Accumulation in Monotonic Loading (T1)
- Setup: SEN-T, 80×80 Q4 mesh, 12 load steps, u_max = 8e-3
- Material: E=210 GPa, ν=0.3, Gc=2.7e-3 GPa·mm, ℓ=0.031
- 6 transfer schemes as proxy for mesh-free storage
- **Figures**:
  - Fig. 1: Phase-field and H-field at final step (exact vs transferred)
  - Fig. 2: H transfer error maps (spatial, 4 schemes)
  - Fig. 3: L2(H) error vs load step (log scale) — the "exponential growth" plot
  - Fig. 4: Error along crack line y=0.5
- **Table 1**: Final errors for all schemes (L2, Linf, tip, growth factor)

### 5.2 Experiment 2: Miehe vs Bourdin (T2)
- Same SEN-T setup, 10 parallel runs
- **Figures**:
  - Fig. 5: THE KEY PLOT — L2 error of carried variable, Miehe (○) vs Bourdin (□)
  - Fig. 6: Spatial error maps, top row (Miehe |ΔH|) vs bottom row (Bourdin |Δd|)
  - Fig. 7: d along crack line, both methods with various transfers
- **Table 2**: Matched-pair comparison (grid 8/16/32, RBF 100)

### 5.3 Experiment 3: Fatigue Cyclic Loading (T3)
- 1D bar with initial crack, 20 cycles, Carrara model
- **Figures**:
  - Fig. 8: Fatigue accumulation ᾱ(x) and degradation f(ᾱ) over cycles
  - Fig. 9: Transfer error in ᾱ and d over cycles
  - Fig. 10: Comparison: standard vs fatigue, exact vs coarse transfer
- **Table 3**: Error growth per cycle, extrapolation to 1000 cycles

### 5.4 Experiment 4 (Optional): 1D Analytical Verification
- Closed-form 1D phase-field solutions confirming mechanisms A/B/C
- Coupled 1D solver: ratio 160–187× confirms 2D result

---

## 6. Discussion (2–3 pages)

### 6.1 Implications for Neural PDE Solver Design
- History transfer is NOT an implementation detail — it's a structural mathematical problem
- Current DRM/PINN approaches (Manav 2024, Goswami 2020) avoid the problem by using FE mesh for history
- Truly mesh-free approaches (future work) MUST address transfer error
- The bounded transfer principle provides actionable guidance

### 6.2 Connection to Incremental Variational Principles
- Ortiz-Stainier (1999) established that all standard dissipative materials admit incremental potentials
- Our work shows that the CHOICE of variational formulation affects mesh-free implementability
- Bourdin vs Miehe: same physics, vastly different transfer robustness
- This is a new dimension of variational formulation selection criteria

### 6.3 Beyond Phase-Field: Plasticity, Viscoelasticity
- Plasticity: ε^p is unbounded, tensorial → worse than H
  - Proposal: carry g(ε^p) or equivalent plastic strain (scalar, bounded if capped)
  - Connection to He et al. (2023) — they sidestep by using FE mesh
- Viscoelasticity: internal stress/strain variables → similar issues
- Fatigue of any kind: cyclic loading amplifies transfer count

### 6.4 Limitations
- Our analysis uses proxy transfer schemes (grid, RBF) rather than actual NN-to-NN transfer
  - Conjecture: NN transfer has similar or worse properties (training noise adds to interpolation error)
- 2D only (3D would amplify crack-tip singularity)
- AT2 model only (AT1 has sharper transitions → possibly worse)
- Staggered scheme only (monolithic would change coupling structure)

---

## 7. Conclusions (1 page)

1. **History variable transfer** is a fundamental bottleneck for mesh-free neural PDE solvers applied to path-dependent materials.
2. **Error grows exponentially** under naive transfer (60,000× in 12 steps for Miehe's H).
3. **Variational formulation choice matters**: Bourdin's bounded irreversibility is 77× more robust than Miehe's unbounded history variable.
4. **Fatigue compounds the problem**: 3 history fields (vs 1), 2 unbounded, and cyclic loading multiplies transfer count.
5. **The bounded transfer principle**: mesh-free solvers should carry only bounded variables, achievable through variational reformulation (Bourdin) or contractive maps (f(ᾱ) for fatigue).

Open questions for future work:
- Rigorous approximation theory for NN-to-NN history transfer
- Adaptive transfer resolution (h-refinement near crack tips)
- Extension to tensorial history variables (elastoplasticity)
- Convergence rates: does bounded transfer give O(h_s^p/ℓ^q)?

---

## Appendices

### A. FEM Implementation Details
- Q4 elements, staggered Newton, convergence criteria
- GPU implementation (PyTorch) for reproducibility

### B. Transfer Scheme Specifications  
- Grid interpolation, RBF kernel/regularization, random sampling details

### C. 1D Analytical Solutions
- Exact d(x) for AT2 model
- Proof of gradient bounds

---

## Reference List (~45–55 refs)

### Core (MUST cite)
- Francfort & Marigo (1998) — variational fracture
- Bourdin, Francfort, Marigo (2000) — regularized phase-field
- Miehe, Welschinger, Hofacker (2010) — staggered scheme + H variable
- Gerasimov & De Lorenzis (2019) — variational consistency, Bourdin > Miehe
- Carrara, Ambati, Alessi, De Lorenzis (2020) — fatigue phase-field
- Ortiz & Stainier (1999) — incremental variational principles
- E & Yu (2018) — Deep Ritz Method
- Raissi, Perdikaris, Karniadakis (2019) — PINNs

### Predecessor (position against)
- He et al. (2023) IJP — hybrid NN-FEM for plasticity, mesh-based history
- Manav et al. (2024) CMAME — DRM phase-field, Bourdin irreversibility
- Goswami et al. (2020) — PINN phase-field fracture
- Abueidda et al. (2022) — DEM for hyperelasticity

### Supporting theory
- Simo & Hughes (2006) — computational inelasticity
- Ambrosio & Tortorelli (1990) — Γ-convergence
- Miehe, Schänzel, Ulmer (2015) — phase-field fatigue framework
- Nguyen (2000) — stability/bifurcation in dissipative systems

### NN approximation theory
- Barron (1993) — approximation by sigmoidal NNs
- Cybenko (1989) — universal approximation
- Mishra & Molinaro (2022) — estimates for PINNs
- Müller & Zeinhofer (2023) — DRM error analysis

### Additional method papers
- Niu & Srivastava (2023) JMPS — NN outputs plastic variables
- Masi & Stefanou (2022) — TANNs
- Flaschel et al. (2022) — EUCLID
- Rezaei et al. (2023) — COMM-PINN

---

## Estimated Length
- Main text: ~18–22 pages (Springer single-column)
- Figures: 10–12
- Tables: 3
- Appendices: ~3 pages
- Total: ~22–25 pages

---

## Author Contributions (CRediT)
- **Haofan [Surname]**: Conceptualization, Methodology, Software, Formal analysis, Investigation, Writing – original draft, Visualization
- **[Supervisor/Co-author]**: Supervision, Writing – review & editing, Funding acquisition
