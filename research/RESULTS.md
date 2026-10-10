# Results on the supplied OPA lattice

All results below use m=8,d=1 and the corrected finite thin-octupole exponential. There is no magnet optimization. The tests cover delta=0,+/-0.01,+/-0.02,+/-0.03. All comparisons are paired on identical physical trajectories within each experiment. Different confirmation experiments use different seeds and must not be compared as if they were the same sample.

## Baseline comparison (256 turns)

345 of 448 randomly phased launches survived. Training and validation used independent initial conditions and 64/128 turns. The baseline coupled solvers hit the 200-iteration limit; their finite iterates are reported without claiming convergence.

| Method | Median common-scale drift | 90th percentile | Median original relative rate |
|---|---:|---:|---:|
| eigen | 35.663 | 796.68 | 0.15908 |
| track_ridge_1 | 35.664 | 796.61 | 0.15964 |
| LS_eigen_seed | 14.06 | 215.64 | 0.36614 |
| LS_CS_seed | 7.8183 | 115.22 | 0.39033 |
| a_box_coupled_fixed | 2.9356 | 40.299 | 0.64759 |
| a_box_coupled_regularized | 2.8784 | 40.214 | 0.64885 |
| graded_coupled | 10.127 | 1185.1 | 0.79749 |
| canonical_graded | 3.6702 | 322.51 | 0.6553 |

The small bounded correction to eigen did not yield a practically meaningful improvement. Replacing the old thin-kick map with the finite exponential changed eigen dramatically; historical pre-fix results are retained separately in opa_m8_legacy. This is a model correction shared by all constructors, not a method-specific advantage.

## General invariant-subspace fit (fresh 512-turn test)

254/336 launches survived. Selected regularization: 1e-06. The fit and selection never used this confirmation set.

| Quantity | Eigen | New fit |
|---|---:|---:|
| Median common-scale maximum drift | 46.9831 | 1.25175 |
| 90th percentile | 996.72 | 14.5506 |
| Worst survivor | 32186.5 | 1391.88 |
| Median original relative rate | 0.172946 | 0.0838134 |
| Median gradient-normalized defect | 0.192998 | 0.139378 |
| 90th-percentile gradient-normalized defect | 0.577788 | 1.77249 |

Paired win fraction (common-scale drift): 91.73%. Median paired error ratio: 0.028927; bootstrap 95% interval [0.022367000798432214, 0.03726781350680727]. This interval quantifies sampling variation in this launch distribution, not uncertainty across different lattices or model errors.

| Delta | Eigen median | Fit median | Eigen p90 | Fit p90 |
|---:|---:|---:|---:|---:|
| -0.03 | 203.27 | 10.114 | 1475.8 | 82.524 |
| -0.02 | 72.748 | 1.8159 | 1607.9 | 12.498 |
| -0.01 | 23.245 | 0.44306 | 382.42 | 14.153 |
| +0.00 | 0.019846 | 0.028051 | 0.27465 | 0.69327 |
| +0.01 | 28.685 | 0.3058 | 150.83 | 3.7119 |
| +0.02 | 166.55 | 3.6026 | 1464.5 | 14.142 |
| +0.03 | 273.57 | 2.4511 | 4754.2 | 10.399 |

## Protected chromatic subspace fit (fresh 512-turn test)

248/336 launches survived. Selected regularization: 0.01. The fit and selection never used this confirmation set.

| Quantity | Eigen | New fit |
|---|---:|---:|
| Median common-scale maximum drift | 34.3515 | 16.6976 |
| 90th percentile | 923.349 | 293.151 |
| Worst survivor | 31357.3 | 4758.95 |
| Median original relative rate | 0.187981 | 0.159296 |
| Median gradient-normalized defect | 0.178183 | 0.178488 |
| 90th-percentile gradient-normalized defect | 0.636771 | 0.596613 |

Paired win fraction (common-scale drift): 77.82%. Median paired error ratio: 0.56644; bootstrap 95% interval [0.5079311636906808, 0.6129711743495085]. This interval quantifies sampling variation in this launch distribution, not uncertainty across different lattices or model errors.

| Delta | Eigen median | Fit median | Eigen p90 | Fit p90 |
|---:|---:|---:|---:|---:|
| -0.03 | 260.1 | 137.54 | 2511.2 | 1303.3 |
| -0.02 | 35.522 | 17.578 | 339.23 | 161.72 |
| -0.01 | 20.319 | 8.155 | 487.45 | 61.931 |
| +0.00 | 0.0091415 | 0.0091415 | 0.40697 | 0.40697 |
| +0.01 | 30.402 | 8.0049 | 834.34 | 84.123 |
| +0.02 | 99.575 | 24.765 | 678.8 | 254.33 |
| +0.03 | 126.38 | 52.836 | 1802.5 | 814.49 |

## Interpretation

The protected construction improves both reported drift metrics while leaving the complete on-momentum polynomial exactly unchanged. The general fit has larger drift improvements but worsens the on-momentum median and the upper tail of the geometric proxy. The protected fit has essentially unchanged median geometric defect, so this is not evidence of improved invariant geometry or dynamic aperture. Some trajectories worsen. The off-momentum absolute errors remain large; better than eigen does not mean a high-quality invariant everywhere.

The successful construction uses near-unit modes with zero quadratic sector, which are excluded from individual normalization by the original eigen selector. Selecting only the original normalizable eigenvectors, or fitting combinations only of those vectors, did not improve the validation criterion. This supports the interpretation that choosing a representative within the nonunique approximate-invariant family matters, rather than eigendecomposition being intrinsically superior to least squares.

## Additional zero-momentum grid

opa_grid contains a 9x9 grid over +/-4 mm, 256 turns, seven momentum offsets, and all baseline methods plus the general kernel fit. It uses the production launch convention: on-momentum orbit plus delta offset. This is an additional test distribution, distinct from the random-phase off-momentum-orbit tests. It is a coarse diagnostic grid, not a publication-resolution FMA. The runner supports larger grids.

## Remaining limits

The rectangular truncation/dispersion issue remains unresolved; d=1 and default A_BOX delta half-width 0.01 limit extrapolation to +/-0.03. The fit is calibrated to a single lattice and physical tracker. It must be rebuilt after changing magnets. No improvement in beam survival, frequency diffusion or dynamic aperture is claimed, because changing an observable does not change particle dynamics. Read AUDIT.md for the complete scope and findings.