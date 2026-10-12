## results/final_kaggle

### Robust policy fidelity

| Model | Direct | Fairness instruction | Extract, then model decides | Same extraction, code decides |
|---|---|---|---|---|
| gemini-3.7-flash | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| gemma-4-31b | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| gpt-oss-120b | 62% [50, 75] | 73% [60, 85] | 100% [100, 100] | 71% [58, 81] |
| gemini-3.1-flash-lite | 48% [35, 60] | 52% [40, 65] | 71% [60, 81] | 69% [58, 79] |
| gpt-5.4-nano | 6% [2, 10] | 4% [0, 8] | 10% [2, 19] | 31% [21, 44] |

### Controls (direct)

| Model | Qualification response | False reject | False accept | Abstain on complete files | Identity swap flips | Same-group name flips | Neutral ID flips | Exact repeat flips | Invalid |
|---|---|---|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 65% | 33% | 1% | 1% | 9% (n=192) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 0% |
| gemini-3.7-flash | 100% | 0% | 0% | 0% | 0% (n=192) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 0% |
| gemma-4-31b | 100% | 0% | 0% | 0% | 0% (n=192) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 0% |
| gpt-5.4-nano | 10% | 5% | 2% | 66% | 14% (n=192) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 0% |
| gpt-oss-120b | 89% | 0% | 0% | 1% | 1% (n=171) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 5% |

### Diagnostic suites (direct)

| Model | Sufficiency families | Appropriate uncertainty | Unnecessary abstention | Relevance families | Realistic families | Audit families |
|---|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 50% | 95% | 16% | n/a | n/a | n/a |
| gemini-3.7-flash | 58% | 100% | 0% | n/a | n/a | n/a |
| gemma-4-31b | 94% | 97% | 0% | n/a | n/a | n/a |
| gpt-5.4-nano | 12% | 95% | 60% | n/a | n/a | n/a |
| gpt-oss-120b | 52% | 87% | 0% | n/a | n/a | n/a |

### Contrasts

| Contrast | Estimate (pp) | 95% interval | p | Holm p | Models |
|---|---|---|---|---|---|
| C1_fairness_minus_direct | +2.5 | [-1.7, +6.3] | 0.257 | 0.456 | 5 |
| C2_transfer | +4.6 | [-0.4, +9.2] | 0.074 | 0.222 | 5 |
| S_qual_response_fairness_minus_direct | +2.0 | [+0.2, +3.6] | 0.028 | nan | 5 |
| C3_policy_code_minus_model_verdict | -2.1 | [-5.0, +1.2] | 0.228 | 0.456 | 5 |
| S_structured_minus_direct | +12.9 | [+9.2, +16.2] | 0.000 | nan | 5 |
| S_policy_code_minus_direct | +10.8 | [+6.7, +15.0] | 0.000 | nan | 5 |

## results/final_cluster

### Robust policy fidelity

| Model | Direct | Fairness instruction | Extract, then model decides | Same extraction, code decides | Extract only, code decides |
|---|---|---|---|---|---|
| gpt-oss-120b | 98% [96, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] | 100% [100, 100] |
| gpt-oss-20b | 96% [94, 98] | 97% [95, 99] | 95% [92, 98] | 99% [98, 100] | 99% [98, 100] |
| mistral-small-3.2-24b | 43% [36, 50] | 43% [36, 49] | 79% [74, 83] | 91% [86, 95] | 90% [86, 94] |
| qwen3-32b | 39% [33, 45] | 36% [30, 43] | 45% [38, 52] | 73% [67, 80] | 71% [65, 78] |
| gemma-3-27b-it | 42% [36, 49] | 41% [35, 47] | 42% [36, 48] | 65% [58, 71] | 65% [58, 71] |
| llama-3.1-8b-instruct | 0% [0, 0] | 0% [0, 0] | 1% [0, 3] | 56% [49, 64] | 41% [34, 47] |

### Controls (direct)

| Model | Qualification response | False reject | False accept | Abstain on complete files | Identity swap flips | Same-group name flips | Neutral ID flips | Exact repeat flips | Invalid |
|---|---|---|---|---|---|---|---|---|---|
| gemma-3-27b-it | 47% | 6% | 38% | 6% | 2% (n=768) | 8% (n=76) | 7% (n=76) | 0% (n=304) | 0% |
| gpt-oss-120b | 100% | 0% | 0% | 0% | 0% (n=768) | 0% (n=76) | 0% (n=76) | 0% (n=304) | 0% |
| gpt-oss-20b | 97% | 0% | 3% | 0% | 0% (n=768) | 0% (n=76) | 0% (n=76) | 2% (n=304) | 0% |
| llama-3.1-8b-instruct | 0% | 0% | 24% | 54% | 6% (n=768) | 5% (n=76) | 4% (n=76) | 0% (n=304) | 0% |
| mistral-small-3.2-24b | 53% | 39% | 4% | 2% | 1% (n=768) | 0% (n=76) | 1% (n=76) | 1% (n=304) | 0% |
| qwen3-32b | 41% | 7% | 41% | 6% | 2% (n=768) | 0% (n=76) | 1% (n=76) | 1% (n=304) | 0% |

### Diagnostic suites (direct)

| Model | Sufficiency families | Appropriate uncertainty | Unnecessary abstention | Relevance families | Realistic families | Audit families |
|---|---|---|---|---|---|---|
| gemma-3-27b-it | 2% | 72% | 25% | 88% | 35% | 42% |
| gpt-oss-120b | 81% | 88% | 0% | 100% | 94% | 92% |
| gpt-oss-20b | 52% | 73% | 0% | 100% | 92% | 92% |
| llama-3.1-8b-instruct | 0% | 89% | 68% | 0% | 2% | 0% |
| mistral-small-3.2-24b | 40% | 94% | 17% | 72% | 50% | 29% |
| qwen3-32b | 25% | 83% | 22% | 75% | 40% | 50% |

### Contrasts

| Contrast | Estimate (pp) | 95% interval | p | Holm p | Models |
|---|---|---|---|---|---|
| C1_fairness_minus_direct | -0.3 | [-1.2, +0.6] | 0.481 | 0.481 | 6 |
| C2_transfer | -1.0 | [-2.0, +0.1] | 0.080 | 0.160 | 6 |
| S_qual_response_fairness_minus_direct | -0.5 | [-1.1, +0.1] | 0.095 | nan | 6 |
| C3_policy_code_minus_model_verdict | +20.5 | [+18.1, +22.8] | 0.000 | 0.000 | 6 |
| S_structured_minus_direct | +7.1 | [+4.9, +9.3] | 0.000 | nan | 6 |
| S_policy_code_minus_direct | +27.6 | [+24.7, +30.5] | 0.000 | nan | 6 |
| S_extract_only_minus_policy_code_same_extraction | -3.0 | [-4.2, -1.8] | 0.000 | nan | 6 |

