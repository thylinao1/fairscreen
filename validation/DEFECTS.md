# Defect register

| Version | Found by | Defect | Correction | Effect on gold labels |
|---|---|---|---|---|
| 1.0.0 -> 1.1.0 (12 Oct 2026, before any confirmatory output) | Blind review, two independent readers (validation/review_readers.json) | Assessment results could be dated in the screening month (Oct 2026), so a careful reader could not tell whether the result predated the 1 October decision date | Assessment dates drawn from the three months before the screening month | none (labels unchanged; the ambiguity is removed) |
| same | same | "1 years" in the bilingual support policy | singular and plural forms | none |
| same | same | Non-qualifying job titles were drawn independently of their descriptions (for example "Events Assistant" who "drove delivery routes") | titles and descriptions paired | none |
| same | same | Whether a waived requirement counts as "clearly met" under the decision rule was implicit | the exception now states that the waived requirement counts as met | none |
| same | same | Missing comma before "or (b)" in alternative requirements | punctuation | none |

Blind review result on dataset 1.0.0 (44 stratified items, readers blind to gold and to all model outputs): both
readers agreed with the gold label on 44 of 44 items. The text-only rule parser (analysis/rule_baseline.py)
agrees with gold on every variant of both versions (2528 confirmatory, 662 development).

The only confirmatory-set model calls made on version 1.0.0 were the Kaggle push-validation run of
`fairscreen-direct` (version 1 of that task, default model). It is superseded by version 2 of the task and is not
used in any analysis.
