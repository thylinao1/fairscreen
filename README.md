# FairScreen

A paired counterfactual benchmark for LLM eligibility screening, built for the DEV x Kaggle Benchmarking Challenge.

Each base family crosses a qualification edit (a stated requirement is crossed or not), an identity edit
(Singapore name or community marker) and a presentation edit (clean or changed layout). A correct system
must change its decision when the qualification crosses the requirement, keep it when only identity changes,
and do both in the changed presentation. Labels: `MEETS`, `DOES_NOT_MEET`, `INSUFFICIENT_EVIDENCE`.

Work in progress. See `protocol.md` for the registered analysis plan.
