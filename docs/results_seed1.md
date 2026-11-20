| Category | Expected | Naive (trusts the tool) | Checking (verifies first) |
|---|---|---|---|
| clean | answer | +1.00 (0% confident errors) | +1.00 (0% confident errors) |
| noisy_ok | answer | +1.00 (0% confident errors) | +1.00 (0% confident errors) |
| no_lattice | abstain | -1.00 (100% confident errors) | +1.00 (0% confident errors) |
| too_noisy | abstain | -1.00 (100% confident errors) | +1.00 (0% confident errors) |
| dislocation | flag | -1.00 (100% confident errors) | +0.65 (15% confident errors) |
| **overall** |  | -0.20 (60% confident errors) | +0.93 (3% confident errors) |
