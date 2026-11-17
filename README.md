# boundary_control_validation

Validation suite for the white paper

**Boundary Control and Exponential Clustering in Four-Dimensional SU(N) Lattice Yang–Mills**
R. Cooperman and A.E. Petrunina V.V., XISOSIS, Vienna, 17 November 2026.

- GitHub: https://github.com/teknoquants/BOUNDARY-CONTROL-AND-EXPONENTIAL-CLUSTERING-IN-FOUR-DIMENSIONAL-SU-N-LATTICE-YANG-MILLS
- DOI: https://doi.org/10.5281/zenodo.24305335
- Authors: Roger Cooperman (ORCID 0009-0009-5011-2135), A.E. Petrunina V.V. (ORCID 0009-0002-4595-736X)

## Run

```
pip install -r requirements.txt
python3 run_all.py
```

Runtime about one minute. Exit code 0 only if every check passes.

| Script | Paper | Checks |
|---|---|---|
| `validate_part1_freezing.py` | Part 1: freezing and heat smoothing do not commute | 4 |
| `validate_part2_onelink.py` | Part 2: one link, β_eff = 2β\|a\| ≤ 12β, Hessian eigenvalues, antipode, monotonicity (numerical) | 7 |
| `validate_part4_gauge.py` | Part 4: H_B = ker D_U*, endpoint-frame modes | 6 |
| `validate_part5_decay.py` | Part 5: lattice counts, Hessian norms, J_α, split time, Gaussian toy | 7 |
| `validate_part6_7.py` | Part 6: (B-1) ⇒ (A); Part 7: negative control ε/16 | 5 |

Total: 29 checks. `common.py` reuses the one-loop heat-flow code of the Challenge A suite (doi:10.5281/zenodo.24247283).
Part 3 (small-block curvature profiles) is a numerical task that the paper leaves open; no script claims it.

## Licence

MIT — see `LICENSE`. Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
