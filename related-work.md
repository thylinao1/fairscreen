# Related work

This note lists the closest published work to FairScreen and states what the benchmark adds. Each work was read from its arXiv text or repository page. The FairScreen row of the table describes the design and claims no result.

## Closest precedents

**Measuring Validity in LLM-based Resume Screening ([arXiv 2602.18550](https://arxiv.org/abs/2602.18550)).** Castleman et al. build, from 186 job descriptions, a resume meeting exactly the required qualifications, variants with k required ones removed or k preferred ones added, and equal-qualification pairs that differ only in a demographic signal. Nine models compare two resumes and may abstain; the paper reports that they "do not reliably abstain when ranking equally-qualified candidates". Overlap: known-gold ordering, identity edits and abstention. That abstention concerns equal pairs. FairScreen's third label applies when the file lacks the evidence to decide a stated requirement.

**Competence-Preserving Resume Perturbations ([arXiv 2609.16517](https://arxiv.org/abs/2609.16517)).** Chen and Xiao render 102 profiles in five forms, keeping 505 of 510 variants after a deterministic check that the evidence is unchanged. For Llama-3.1-8B, 29.6 percent of matched pairwise decisions flip, and a phrase-matching control has flip rate 0.000 by construction. Overlap: the presentation axis and a deterministic comparison. The paper has no identity edits. FairScreen runs its deterministic executor on the model's own extraction, so the comparison includes the extraction step.

**Karvonen and Marks ([arXiv 2506.10922](https://arxiv.org/abs/2506.10922)).** Anti-bias prompts that work in simple settings stop working once realistic context is added. Interview-rate differences by race and gender reach 12 percent, and editing internal concept directions reduces them. Overlap: whether a mitigation transfers when the situation changes. The change there is job and company framing. FairScreen changes the layout and wording of the same evidence and has a gold label for every decision.

**FAIRE ([arXiv 2504.01420](https://arxiv.org/abs/2504.01420)).** Wen et al. edit names and add generated descriptions of racial experiences to real resumes from a public Kaggle dataset; five models score and rank them. The paper reports that "every model had some bias gap". Overlap: counterfactual identity edits. The baseline resumes have no known qualification order, and the edits also change experience text, so name and experience are partly confounded. FairScreen's identity edit changes only identity fields.

**PopResume ([arXiv 2603.22714](https://arxiv.org/abs/2603.22714)).** Yu, Park and Moon sample 60.8K resumes from US population statistics and separate the effect of gender or race into a path through education and experience and a path through name and address. In 49 of 80 cases with no explicit demographic text, a significant direct effect remained. Overlap: presentation variation (three formats, image input). PopResume estimates qualification as a mediator, and FairScreen sets it as a labelled factor. PopResume is the better-suited design for causal estimates on population-realistic resumes.

**FairFund-Bench ([repository](https://github.com/martinlukk/fairfund-bench)).** This benchmark covers 600 aid requests, not hiring, and reports demographic bias, deservingness alignment and consistency across tasks and contexts for 14 models. Its README describes no mitigation and no abstain label. Overlap: reporting bias, an alignment score and consistency together. We read the README only.

## Comparison

Y = yes, P = partial, N = no.

| Work | Counterfactual identity edits | Known-gold qualification ordering | Insufficient-evidence label | Mitigation tested | Transfer across presentation | Deterministic policy execution on the model's own extraction |
|---|---|---|---|---|---|---|
| Validity (2602.18550) | Y | Y | P (equal pairs) | N | N | N |
| Competence-Preserving (2609.16517) | N | Y | N | N | Y | P (control on the text) |
| Karvonen and Marks (2506.10922) | Y | N | N | Y | P (context shift) | N |
| FAIRE (2504.01420) | Y | N | N | N | N | N |
| PopResume (2603.22714) | P | N | N | N | P | N |
| FairFund-Bench | Y | P | N | N | P | N |
| FairScreen (design) | Y | Y | Y | Y | Y | Y |

## What is new here

Every column has a precedent. We are not aware of an existing benchmark that combines, in one factorial design: a qualification edit that crosses a stated threshold, with a gold label from an explicit policy; identity edits for a Singapore locale (names and community markers); a change in the presentation of the same evidence; an INSUFFICIENT_EVIDENCE gold label; a fairness-instruction arm tested on clean and changed presentations; and deterministic policy code applied to the extraction behind the model's own verdict, which lets an error be assigned to extraction or to aggregation. For Singapore, Tan et al. ([arXiv 2603.05189](https://arxiv.org/abs/2603.05189)) varied sociocultural markers across 18 LLMs. Our searches were limited.

## Regulatory context

New York City Local Law 144 bars an employer from using an automated employment decision tool unless information about its bias audit is public ([DCWP page](https://www.nyc.gov/site/dca/about/automated-employment-decision-tools.page)). The adopted rule defines the impact ratio as the selection rate for a category divided by that of the most selected category, computed for sex, race/ethnicity and intersectional categories ([rule text](https://rules.cityofnewyork.us/wp-content/uploads/2023/04/DCWP-NOA-for-Use-of-Automated-Employment-Decisionmaking-Tools-2.pdf)). That metric compares aggregate selection rates. It does not test whether one application receives the same decision when only an identity signal changes. We found no statement that it requires a paired test. Annex III of the EU AI Act lists systems "to analyse and filter job applications, and to evaluate candidates" among high-risk uses ([Annex III](https://artificialintelligenceact.eu/annex/3/)). A FairScreen score is not a bias audit or a compliance finding under either regime.
