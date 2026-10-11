# FairScreen dataset card

Version 1.1.0. Files: `data/dev/`, `data/final/`, `data/manifest.json`.

## What the data is

FairScreen contains fictional job applicants and fictional eligibility policies. Each application is rendered as text from a hidden canonical record. Gold labels are computed from the record and the policy by two implementations (`src/fairscreen/policy.py` and `src/fairscreen/runtime.py::replay`) that agree on every generated variant. A text-only rule parser (`analysis/rule_baseline.py`) reads only the rendered text and reproduces the gold label on every variant, which checks that each rendering states its record (it was written with knowledge of the templates and is not a competitor). Two blind readers agreed with the gold label on 44 of 44 sampled items of version 1.0.0 (`validation/DEFECTS.md`). No model defines a label.

Labels are `MEETS`, `DOES_NOT_MEET` and `INSUFFICIENT_EVIDENCE`. A policy is a conjunction of requirements, each TRUE, FALSE or UNKNOWN. Any FALSE gives `DOES_NOT_MEET`, all TRUE gives `MEETS`, and otherwise the label is `INSUFFICIENT_EVIDENCE`.

## Suites and counts

| Suite | Families (dev / final) | Variants (dev / final) |
|---|---|---|
| Core | 48 / 192 | 438 / 1728 |
| Evidence sufficiency | 16 / 48 | 64 / 192 |
| Relevance | 8 / 32 | 32 / 128 |
| Realistic documents | 8 / 48 | 32 / 192 |
| Audit framing | 8 / 24 | 96 / 288 |
| Total | 88 / 344 | 662 / 2528 |

A core family crosses qualification state (a requirement crossed or not), identity (A or B) and presentation (clean or changed), giving eight variants, and some families add control variants. Sufficiency, relevance and realistic families have four variants each, and audit families have twelve. MEETS and DOES_NOT_MEET are equally frequent in every suite, and INSUFFICIENT_EVIDENCE is half of the sufficiency variants.

## Identity signals

The locale is Singapore. Names are Chinese, Malay and Indian, each female and male, with eight names per group (four per split) in local conventions: Chinese surname first, Malay with bin or binte and a father's name, Indian with s/o or d/o and a father's name. The indirect marker is a volunteering line for CDAC (Chinese), Yayasan MENDAKI (Malay) or SINDA (Indian), written under a neutral applicant ID in place of a name.

Names and markers signal a perceived group to a reader. They say nothing about the ethnicity, gender, religion or qualifications of any real person, including people who share these names. All applicants are fictional. Malay names may also signal religion.

## Locale and policy assumptions

Experience is counted to a screening date of 1 October 2026. Employers are invented, and education lines name Singapore institutions. Policies are benchmark specifications and make no claim about how these jobs should be filled. Each has two requirements, drawn from experience thresholds, assessment results, certificate validity, a degree-or-experience alternative, an exception (certificate waived after five years) and a language requirement. Threshold crossings keep a margin of at least five months. In the relevance suite the name group matches the language skill (Mandarin with Chinese names, Malay with Malay names, Tamil with Indian names), so skill and perceived group co-occur by construction.

## Development and held-out split

Development uses the classic layout, policy phrasing 1, the first four names of each pool and four occupations (backend developer, customer support, payroll operations, warehouse coordination). The confirmatory split (`final`) uses the modern layout, policy phrasing 2, the last four names of each pool and eight occupations. The four held-out occupations are data analyst, laboratory support, technical writing and bilingual support; the bilingual support policy also appears in the development relevance suite.

## Intended use

Testing policy fidelity (whether a decision follows the stated policy applied to the evidence in the file) and paired invariance (whether the decision stays the same when only identity or only presentation changes) of screening configurations.

## Out-of-scope use

Certifying a hiring system as fair or compliant, predicting job performance, establishing legal compliance, and making statements about real people or groups.

## Known limitations

- The name lists are our own choice of common forms. They were not validated by human raters or against a frequency table.
- Malay and Indian names carry a patronymic marker and are longer than Chinese names, so a name swap changes more than perceived group.
- The Indian group uses mainly South Indian names, and Indian Singaporeans are more varied.
- The text is English only, from one locale and one screening date.
- Core documents are short and templated.
- Threshold margins were chosen to be unambiguous, so the data says nothing about near-threshold decisions.

## Licence

Data: CC BY 4.0. Code: MIT.

## Regeneration

```
PYTHONPATH=src python -m fairscreen.build_dataset
PYTHONPATH=src pytest tests
python analysis/rule_baseline.py
```

`data/manifest.json` records the SHA-256 digest of each data file.
