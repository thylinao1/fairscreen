"""Template-aware rule parser: reads only the rendered application text, extracts the facts with regular
expressions, and applies the same policy code. It is not a fair competitor (it was written knowing the
templates); it checks that every rendering expresses its canonical record, and gives the 'no LLM' reference row.

python analysis/rule_baseline.py --split final
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fairscreen import runtime as RT  # noqa: E402

MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
NOW = 2026 * 12 + 9
DATE = r"([A-Z][a-z]+) (\d{4})"
RANGE = re.compile(DATE + r" to (?:" + DATE + r"|present)")


def mi(mon, year):
    return int(year) * 12 + MON[mon[:3].lower()]


def role_chunks(text):
    """Split the text into chunks, each starting at a line that carries a role date range or 'dates not stated'."""
    lines = text.split("\n")
    chunks, cur = [], None
    for ln in lines:
        if RANGE.search(ln) and not ln.lower().startswith(("- a reference", "a reference")) or "dates not stated" in ln:
            if "reference check" in ln.lower():
                continue
            cur = [ln]
            chunks.append(cur)
        elif cur is not None and ln.startswith("  "):
            cur.append(ln)
        else:
            cur = None
    return [" ".join(c) for c in chunks]


def notes(text):
    return [ln for ln in text.split("\n") if "conflict has not been resolved" in ln]


def extract(v):
    text = v["application_text"]
    spec = v["policy_spec"]
    nts = notes(text)
    facts = {}
    for k in v["fact_keys"]:
        if k.startswith("years:"):
            skill = k[6:]
            intervals, undated = [], False
            for ch in role_chunks(text):
                if skill.lower() not in ch.lower():
                    continue
                m = RANGE.search(ch)
                if not m:
                    undated = True
                    continue
                s = mi(m.group(1), m.group(2))
                e = NOW if m.group(3) is None else mi(m.group(3), m.group(4))
                intervals.append((s, e))
            intervals.sort()
            tot, cs, ce = 0, None, None
            for s, e in intervals:
                if cs is None:
                    cs, ce = s, e
                elif s <= ce:
                    ce = max(ce, e)
                else:
                    tot += ce - cs
                    cs, ce = s, e
            if cs is not None:
                tot += ce - cs
            conflict = any("start date of the" in n for n in nts)
            facts[k] = None if (undated or conflict) else round(tot / 12, 2)
        elif k.startswith("credential:"):
            name = k[11:]
            if "Details available on request" in text:
                facts[k] = "UNKNOWN"
            else:
                line = next((ln for ln in text.split("\n") if name in ln and "registry" not in ln), "")
                if not line:
                    facts[k] = "NOT_HELD"
                elif "valid until" in line:
                    facts[k] = "VALID"
                elif "expired" in line:
                    facts[k] = "EXPIRED"
                elif "revoked" in line:
                    facts[k] = "REVOKED"
                else:
                    facts[k] = "UNKNOWN"
            if any(name in n for n in nts):
                facts[k] = "UNKNOWN"
        elif k.startswith("assessment:"):
            name = k[11:]
            line = next((ln for ln in text.split("\n") if name.lower() in ln.lower() and "vendor" not in ln), "")
            low = line.lower()
            if "passed" in low:
                facts[k] = "PASSED"
            elif "failed" in low:
                facts[k] = "FAILED"
            elif "not taken" in low or "has not taken" in low:
                facts[k] = "NOT_TAKEN"
            else:
                facts[k] = "UNKNOWN"
            if any(name in n for n in nts):
                facts[k] = "UNKNOWN"
        elif k == "degree_level":
            if "Details to be provided" in text:
                facts[k] = "UNKNOWN"
            else:
                lv = "NONE"
                for pat, val in (("PhD", "PHD"), ("Master of", "MASTER"), ("Bachelor of", "BACHELOR"),
                                 ("Diploma in", "DIPLOMA"), ("Higher Nitec", "NITEC")):
                    if pat in text:
                        lv = val
                        break
                facts[k] = lv
        elif k == "degree_field":
            if "Details to be provided" in text:
                facts[k] = "UNKNOWN"
            else:
                m = re.search(r"(?:Higher Nitec|Diploma|Bachelor of Science|Master of Science|PhD) in ([A-Za-z ]+?)(?:,| from|$|\n)", text)
                fld = m.group(1).strip() if m else "NONE"
                facts[k] = fld if fld in RT.degree_fields(spec) else "OTHER"
                if any("field of study" in n for n in nts):
                    facts[k] = "UNKNOWN"
        elif k.startswith("language:"):
            lang = k[9:]
            if re.search(r"Languages[^\n]*\n(- )?Not stated", text) or "LANGUAGES (all)\nNot stated" in text:
                facts[k] = "UNKNOWN"
            else:
                m = re.search(lang + r" \((native|professional working proficiency|conversational|basic)\)", text)
                facts[k] = {"native": "NATIVE", "professional working proficiency": "PROFESSIONAL",
                            "conversational": "CONVERSATIONAL", "basic": "BASIC"}[m.group(1)] if m else "NONE"
                if any("language test report" in n for n in nts):
                    facts[k] = "UNKNOWN"
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="final")
    a = ap.parse_args()
    vs = [json.loads(l) for l in open(f"data/{a.split}/variants.jsonl")]
    bad = []
    for v in vs:
        f = extract(v)
        d = RT.replay(v["policy_spec"], f)
        if d != v["gold_decision"]:
            bad.append((v["variant_id"], v["gold_decision"], d, f, v["gold_facts"]))
    print(f"{a.split}: rule parser agrees with gold on {len(vs) - len(bad)}/{len(vs)} variants")
    for b in bad[:8]:
        print(b)
    return len(bad)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
