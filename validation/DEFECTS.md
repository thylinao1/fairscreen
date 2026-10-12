# Defect register

| Version | Found by | Defect | Correction | Effect on gold labels |
|---|---|---|---|---|
| 1.0.0 -> 1.1.0 (12 Oct 2026, before any confirmatory output) | Blind review by two separate Claude Sonnet sessions, given only the policy and application text (validation/review_readers.json) | An assessment result could be dated in the screening month (Oct 2026), so a careful reader could not tell whether it came before the 1 October decision date | Assessment dates drawn from the three months before the screening month | none (labels unchanged; the ambiguity is removed) |
| same | same | The bilingual support policy contained "1 years" | singular and plural forms | none |
| same | same | Non-qualifying job titles were drawn independently of their descriptions (for example, an "Events Assistant" who "drove delivery routes") | titles and descriptions paired | none |
| same | same | The decision rule did not say whether a waived requirement counts as "clearly met" | the exception now states that a waived requirement counts as met | none |
| same | same | Missing comma before "or (b)" in alternative requirements | punctuation | none |

Blind review result on dataset 1.0.0 (44 stratified items): the two readers, separate Claude Sonnet sessions and not people, were blind to gold labels, code and all model outputs. Both agreed with the gold label on 44 of 44 items. The text-only rule parser (analysis/rule_baseline.py) agrees with gold on every variant of both versions (2528 confirmatory, 662 development).

The only confirmatory-set model calls on version 1.0.0 came from the Kaggle push-validation run of `fairscreen-direct` (version 1 of that task, default model). Version 2 of the task supersedes this run, and no analysis uses it.
