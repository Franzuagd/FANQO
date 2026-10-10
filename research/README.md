# Research comparison: physical tracking versus polynomial-map invariance

This experiment is separate from the production workflow. It does not change the user's configuration or constructors. `track_fit.py --config ../user/general_config.py --order 8` uses an explicit local configuration; without `--config` it uses the branch reference lattice. Results on that lattice are not evidence about a different OPA lattice.

The protocol is fixed before test evaluation:

* Same physical ring, corrected magnet strengths, polynomial basis and order for all methods.
* Random horizontal and vertical betatron phases, with nonzero momenta. Three momentum offsets: 0 and +/-0.01. Uniform position amplitudes between 0.15 mm and `--extent` (default 4 mm).
* Separate random initial conditions for training (64 turns), validation (128 turns), and test (default 256 turns). Default 48 launches per delta for training and validation, 96 for test. Entire trajectories belong to one split; individual turns are never split across training and test.
* Lost trajectories are counted and excluded identically for all constructors. Conclusions apply to surviving trajectories, not dynamic aperture improvements.
* Six ridge strengths are selected by validation 90th-percentile maximum drift, subject to RMS initial-value change <=0.25 in the fixed CS-action scale. Eigen itself is an eligible choice. Test data are generated only after selection is frozen.
* Main metric is maximum multi-turn drift divided by the same initial CS-action scale for all candidates. The original per-turn relative drift is also reported. Full eigen quadratic coefficients are fixed in the fitted candidate, preventing arbitrary rescaling or the zero polynomial.
* Report median, 90th percentile, worst case, fraction of paired wins, per-delta statistics and a paired bootstrap interval. The initial-value constraint helps prevent an artificially flat observable, but does not prove positivity or global action monotonicity.

The experiment includes the expensive coupled methods, both structured methods, CS-seeded LS and a full-residual LS solve with eigen's exact quadratic seed. The latter isolates differences in feasible sets. Basis conditioning, regularization and minimizer selection can still cause different tracking performance even for identical quadratic constraints.

The fitted method is a correction `I = I_eigen + sum_j u_j phi_j`, using only higher-sector monomials with nonzero transverse degree. For tracked points define `D_ij=(phi_j(z_i(k))-phi_j(z_i(0)))/Sx(z_i(0))`. Solve

    min_u ||D_free u + D c_eigen||_2^2/N + lambda ||S u||_2^2,

where S contains empirical initial-value feature scales. This convex problem fits the physical tracking map directly. It is related to finite-dictionary Koopman methods, but it does not diagonalize an empirical operator.

Since u=0 is feasible, the exact minimizer has training squared drift no greater than eigen (the regularizer is nonnegative and vanishes at zero). This does NOT imply better validation/test maximum drift. Validation rejects corrections that do not improve its criterion.

Construction costs must include tracking, feature assembly, and regularization selection, not just the final linear solve. The new method has substantially more information and a larger computational budget than eigen. Its proper claim is improved empirical invariance for a fixed machine and domain, if the held-out results support that claim; it is not a universally superior cheap algebraic constructor.

## Mathematical explanation

Let U be the pullback of the physical tracking map and T the truncated polynomial operator. Small `(T-I)c` only controls the retained formal residual. The actual defect is `(U-I)I_c = (T-I)I_c + (U-T)I_c`. The second term includes truncation and differences between tracking and the polynomial model. Over N turns, drift is the telescoping sum of one-turn physical defects along the trajectory. Neither a small coefficient residual nor a zero bracket with an auxiliary invariant bounds that second term without additional estimates.

When LS fixes CS quadratic coefficients but eigen selects a different quadratic block, eigen need not belong to the LS feasible set. If the identical eigen quadratic block is fixed and the full weighted residual is minimized, eigen IS feasible. Therefore exact unregularized LS cannot have a larger minimum weighted map residual than eigen. It may nevertheless have worse physical tracking drift: its nullspace representative or higher-order coefficients can amplify the unmodelled defect. A regularized objective trades residual against coefficient size and does not satisfy the unregularized residual comparison in general.

A zero bracket expresses mutual involution, not invariance under a specified map. Even two commuting CS actions need not be preserved by a nonlinear ring. Adding a bracket penalty is therefore not a direct substitute for minimizing physical invariance error.

## Primary literature

* Williams, Kevrekidis and Rowley, *A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition*, J. Nonlinear Science 25 (2015), 1307–1346. https://arxiv.org/abs/1408.4408 . Supports using snapshot pairs and a dictionary to approximate observable evolution.
* Zhang et al., *Evaluating the accuracy of the dynamic mode decomposition*, J. Computational Dynamics 7 (2020), 35–56. https://doi.org/10.3934/jcd.2020002 . Evaluates accuracy at the eigenfunction level rather than relying solely on fitted eigenvalues.
* Liu and Tegmark, *Machine Learning Conservation Laws from Trajectories*, Physical Review Letters 126 (2021), 180604. https://doi.org/10.1103/PhysRevLett.126.180604 . An example of learning conserved quantities from trajectory geometry; not the same algorithm implemented here.

These references motivate the approach, not a claim of novelty or proof that it outperforms eigen for FANQO.

## Supplied OPA lattice and invariant-subspace fitting

`lattice_opa.py` is a copy of the user's supplied lattice. `opa_config.py` combines it with the branch settings. The seven-offset experiment is:

```bash
python research/track_fit.py --config research/opa_config.py --order 8 --train 32 --test 64 --seed 4301 --deltas 0 -.01 .01 -.02 .02 -.03 .03 --output research/opa_m8
python research/select_kernel.py --source research/opa_m8 --config research/opa_config.py --output research/opa_kernel_confirmation --seed 99103 --test 48 --turns 512
```

The second stage includes all near-unit eigenspace directions satisfying `abs(lambda(T-I)) <= 1e-8`, including directions with zero quadratic sector that the original selector cannot normalize individually. Real and imaginary parts form a real basis. The basis is scaled by sampled initial-value norms. A numerical nullspace enforces zero changes to the complete eigen quadratic block and pure-delta coefficients. Then a small SVD least-squares problem fits physical drift within that subspace, penalizing changes in initial values.

This is a least-squares construction in an approximate invariant subspace, initialized at eigen. It is different from both picking a single eigenvector and allowing all polynomial coefficients to vary. The five regularization values are selected on validation trajectories only, with eigen available as fallback. A candidate must improve validation median absolute-normalized drift and the original relative metric before competing on the 90th percentile. A fresh random seed is used for 512-turn confirmation. The technique is related to selecting a useful representative from the nonunique approximate invariant family; no universal dominance theorem is claimed.

`select_eigen.py` (selection among the original normalizable eigenvectors) and `select_subspace.py` (combinations of only those normalizable modes) record unsuccessful alternatives. They selected the original eigen result. The crucial additional directions in `select_kernel.py` include modes whose quadratic sector vanishes.

`geometry_check.py` provides a post-selection first-order normal-displacement proxy using the initial full transverse gradient in CS coordinates. This helps detect flat-observable artifacts, but is not a global positivity/monotonicity proof. All negative trials remain documented.

## Single-file local runner

From the repository root, after installing FANQO with `python -m pip install -e ".[tracking,dev]"`, run:

```bash
python user/run_invariance_research.py fit --config research/opa_config.py --order 8 --train 32 --test 64 --seed 4301 --deltas 0 -.01 .01 -.02 .02 -.03 .03 --output research/local_fit
python user/run_invariance_research.py select --config research/opa_config.py --source research/local_fit --output research/local_protected --protect-on-momentum --seed 99104 --test 48 --turns 512
python user/run_invariance_research.py grid --config research/opa_config.py --source research/local_fit --selected research/local_protected --output research/local_grid --points 9 --turns 256 --extent 4
```

The protected option fixes the complete delta-zero polynomial exactly. Replace the configuration path to use your own settings. Refit after changing the lattice. The existing production runner and constructor defaults remain in place. Large trajectory caches are regenerated locally; results and selected coefficients are retained.
