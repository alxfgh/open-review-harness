# Canary results

Synthetic papers with one planted error each, plus one with none. They test the harness: a failure here should be investigated before any result on real papers is trusted.

Judge: gpt-6.1-sol. Prompts: `3d0fdfd5edaf`. Cost: $0.10.

## gpt-6-luna: all passed

| Canary | Arm | Result | Findings |
|---|---|---|---|
| sample_totals | zero_shot | Passed | 2 |
| sample_totals | harness | Passed | 1 |
| false_induction | zero_shot | Passed | 1 |
| false_induction | harness | Passed | 2 |
| contradictory_stats | zero_shot | Passed | 1 |
| contradictory_stats | harness | Passed | 1 |
| injection_units | zero_shot | Passed | 2 |
| injection_units | harness | Passed | 3 |
| clean_control | zero_shot | Passed | 0 |
| clean_control | harness | Passed | 0 |

## gpt-6.1-terra: all passed

| Canary | Arm | Result | Findings |
|---|---|---|---|
| sample_totals | zero_shot | Passed | 1 |
| sample_totals | harness | Passed | 1 |
| false_induction | zero_shot | Passed | 1 |
| false_induction | harness | Passed | 1 |
| contradictory_stats | zero_shot | Passed | 1 |
| contradictory_stats | harness | Passed | 1 |
| injection_units | zero_shot | Passed | 3 |
| injection_units | harness | Passed | 4 |
| clean_control | zero_shot | Passed | 0 |
| clean_control | harness | Passed | 0 |
