"""Guards on analysis/analyze.py: the expected-family check and the registered numbers.

A scheduled family with no response records must count as failed, never vanish from the denominator, and the
registered outputs for the complete saved runs must stay byte-identical.
"""
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE_CELLS = 8  # qualification x identity x presentation


@pytest.fixture(scope="module")
def A():
    """analysis/analyze.py as a module (it is a script, not a package); data paths in it are relative to the repo root."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        spec = importlib.util.spec_from_file_location("analyze_under_test", ROOT / "analysis" / "analyze.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def final(A):
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        vs, fams = A.load_data("final")
    finally:
        os.chdir(cwd)
    core_ids = [f["family_id"] for f in fams if f["suite"] == "core"]
    return vs, fams, core_ids


def perfect_records(vs, fams, family_ids, recipe="direct", model="probe"):
    """Gold-answer records (rep 0, status json) for every variant of the listed families."""
    by_id = {f["family_id"]: f for f in fams}
    return [{"model": model, "recipe": recipe, "variant_id": v, "family_id": fid, "suite": by_id[fid]["suite"], "rep": 0,
             "status": "json", "decision": vs[v]["gold_decision"], "gold": vs[v]["gold_decision"], "usage": {}}
            for fid in family_ids for v in by_id[fid]["variant_ids"]]


# ------------------------------------------------------------------ the missing-family case
def test_one_perfect_family_against_the_192_set_is_not_100_percent(A, final):
    vs, fams, core_ids = final
    assert len(core_ids) == 192
    view = A.View(perfect_records(vs, fams, core_ids[:1]))
    m = A.metrics(view, vs, fams, expected_core=core_ids)
    assert m["robust"][0] == pytest.approx(1 / 192)
    assert m["robust"][0] < 1.0
    assert m["coverage"] == pytest.approx(1 / 192)
    assert m["clean_quartet"][0] == pytest.approx(1 / 192)
    assert m["changed_quartet"][0] == pytest.approx(1 / 192)
    assert m["accuracy"] == pytest.approx(1 / 192)
    assert len(m["_fam_pass"]) == 192 and sum(m["_fam_pass"].values()) == 1
    # a family with no records cannot be a "completed item" either
    assert m["robust_completed_items"][1] == 1


def test_kaggle_slice_missing_family_lowers_score_and_coverage(A, final):
    vs, fams, core_ids = final
    first48 = A.resolve_expected_core(48, "final", "records", fams)
    assert first48 == frozenset(core_ids[:48])
    full = A.metrics(A.View(perfect_records(vs, fams, core_ids[:48])), vs, fams, expected_core=first48)
    assert full["robust"][0] == 1.0 and full["coverage"] == 1.0
    short = A.metrics(A.View(perfect_records(vs, fams, core_ids[:47])), vs, fams, expected_core=first48)
    assert short["robust"][0] == pytest.approx(47 / 48)
    assert short["coverage"] == pytest.approx(47 / 48)


def test_complete_run_scores_the_same_with_or_without_the_expected_set(A, final):
    vs, fams, core_ids = final
    view = A.View(perfect_records(vs, fams, core_ids[:48]))
    legacy = A.metrics(view, vs, fams)
    explicit = A.metrics(view, vs, fams, expected_core=core_ids[:48])
    for key in ("robust", "robust_completed_items", "clean_quartet", "changed_quartet", "accuracy", "coverage", "invalid",
                "qual_response", "_fam_pass", "_clean", "_changed", "_qual_by_fam"):
        assert legacy[key] == explicit[key], key


def test_default_without_expected_set_still_scores_only_families_present(A, final):
    """Documented legacy default, kept for the exploratory scripts that score subset runs on purpose."""
    vs, fams, core_ids = final
    m = A.metrics(A.View(perfect_records(vs, fams, core_ids[:1])), vs, fams)
    assert m["robust"][0] == 1.0 and m["coverage"] == 1.0 and list(m["_fam_pass"]) == core_ids[:1]


def test_expected_set_with_a_non_core_id_is_refused(A, final):
    vs, fams, core_ids = final
    suff = next(f["family_id"] for f in fams if f["suite"] == "sufficiency")
    with pytest.raises(ValueError):
        A.metrics(A.View(perfect_records(vs, fams, core_ids[:1])), vs, fams, expected_core=core_ids[:2] + [suff])


# ------------------------------------------------------------------ expected-set resolution and audit
def test_resolve_expected_core(A, final):
    _, fams, core_ids = final
    assert A.resolve_expected_core("auto", "final", "local", fams) == frozenset(core_ids)
    assert A.resolve_expected_core("auto", "final", "records", fams) == frozenset(core_ids[:A.KAGGLE_CORE_SLICE])
    assert A.KAGGLE_CORE_SLICE == 48
    assert A.resolve_expected_core("auto", "dev", "local", fams) is None
    assert A.resolve_expected_core("observed", "final", "local", fams) is None
    assert A.resolve_expected_core("all", "final", "records", fams) == frozenset(core_ids)
    assert A.resolve_expected_core("10", "final", "local", fams) == frozenset(core_ids[:10])
    with pytest.raises(ValueError):
        A.resolve_expected_core("everything", "final", "local", fams)
    with pytest.raises(ValueError):
        A.resolve_expected_core("0", "final", "local", fams)


def test_audit_lists_missing_incomplete_unexpected_and_duplicate_ids(A, final):
    vs, fams, core_ids = final
    expected = frozenset(core_ids[:48])
    recs = perfect_records(vs, fams, core_ids[:46])                       # 46 and 47 are missing
    recs = [r for r in recs if r["variant_id"] != A.core_cells(core_ids[3])[0]]   # one cell of family 3 is absent
    recs += perfect_records(vs, fams, [core_ids[100]])                    # a core family outside the slice
    recs.append(dict(recs[0], variant_id="core-final-9999/q=hi-id=A-ctx=clean", family_id="core-final-9999"))  # unknown family
    recs.append(dict(recs[0], variant_id=core_ids[0] + "/q=zz"))          # unknown variant of a known family
    audit = A.audit_families(A.View(recs), vs, fams, expected, duplicates=[f"{recs[0]['variant_id']}"])
    assert audit["expected"] == 48
    assert audit["missing"] == [core_ids[46], core_ids[47]]
    assert audit["incomplete"] == [core_ids[3]]
    assert set(audit["unexpected"]) == {core_ids[100], "core-final-9999", core_ids[0] + "/q=zz"}
    assert audit["duplicate"] == [recs[0]["variant_id"]]
    text = A.format_audit("probe | direct", audit)
    assert text.startswith("WARNING [probe | direct]")
    assert core_ids[46] in text and "missing (2" in text and "duplicate (1" in text and "unexpected" in text


def test_audit_is_silent_for_a_complete_run(A, final):
    vs, fams, core_ids = final
    audit = A.audit_families(A.View(perfect_records(vs, fams, core_ids[:48])), vs, fams, frozenset(core_ids[:48]))
    assert not A.audit_has_issues(audit)


def test_dedup_reports_duplicates_and_keeps_the_first_non_transport_record(A):
    base = {"model": "m", "recipe": "direct", "variant_id": "core-final-0000/x", "rep": 0}
    recs = [dict(base, status="transport_error", decision=None), dict(base, status="json", decision="MEETS"),
            dict(base, status="json", decision="DOES_NOT_MEET"), dict(base, variant_id="core-final-0000/y", status="json")]
    kept, dups = A.dedup_records(recs)
    assert dups == {("m", "direct", "core-final-0000/x", 0): 3}
    assert [r["decision"] for r in kept if r["variant_id"].endswith("/x")] == ["MEETS"]
    assert len(kept) == 2


# ------------------------------------------------------------------ end to end: the script on a one-family probe
def run_main(A, monkeypatch, argv):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["analyze.py"] + argv)
    A.main()


def test_script_scores_a_one_family_probe_as_failing_and_prints_a_warning(A, final, tmp_path, monkeypatch, capsys):
    vs, fams, core_ids = final
    local = tmp_path / "local" / "probe-model"
    local.mkdir(parents=True)
    with open(local / "direct.jsonl", "w") as fh:
        for r in perfect_records(vs, fams, core_ids[:1], model="ignored"):
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "out"
    run_main(A, monkeypatch, ["--local", str(tmp_path / "local"), "--split", "final", "--out", str(out)])
    err = capsys.readouterr().err
    row = next(iter(csv.DictReader(open(out / "main_table.csv"))))
    assert float(row["robust"]) == pytest.approx(1 / 192) and float(row["coverage"]) == pytest.approx(1 / 192)
    assert "WARNING [probe-model | direct]" in err and "missing (191" in err and core_ids[1] in err
    check = json.load(open(out / "metrics.json"))["probe-model|direct"]["family_check"]
    assert len(check["missing"]) == 191 and check["expected"] == 192

    # the explicit opt-out for pilot runs keeps the old denominator
    out2 = tmp_path / "out2"
    run_main(A, monkeypatch, ["--local", str(tmp_path / "local"), "--split", "final", "--out", str(out2),
                              "--expected-core", "observed"])
    row2 = next(iter(csv.DictReader(open(out2 / "main_table.csv"))))
    assert float(row2["robust"]) == 1.0


def test_script_expects_48_families_for_kaggle_records(A, final, tmp_path, monkeypatch, capsys):
    vs, fams, core_ids = final
    task = tmp_path / "records" / "fairscreen-direct"
    task.mkdir(parents=True)
    with open(task / "probe-model.jsonl", "w") as fh:
        for r in perfect_records(vs, fams, core_ids[:47], model="ignored"):   # family 47 never ran
            fh.write(json.dumps(r) + "\n")
    out = tmp_path / "out"
    run_main(A, monkeypatch, ["--records", str(tmp_path / "records"), "--split", "final", "--out", str(out)])
    err = capsys.readouterr().err
    row = next(iter(csv.DictReader(open(out / "main_table.csv"))))
    assert float(row["robust"]) == pytest.approx(47 / 48) and float(row["coverage"]) == pytest.approx(47 / 48)
    assert "missing (1;" in err and core_ids[47] in err and "48 expected core families" in err


# ------------------------------------------------------------------ registered numbers do not move
REGISTERED = ("main_table.csv", "main_table.md", "contrasts.json", "metrics.json")


def rerun_and_compare(tmp_path, source_flag, source_dir, reference):
    ref = ROOT / reference
    if not (ref / "main_table.csv").exists() or not (ROOT / source_dir).exists():
        pytest.skip(f"{reference} or {source_dir} not present")
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "analysis/analyze.py", source_flag, source_dir, "--split", "final", "--out", str(out)],
                          cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert "WARNING" not in done.stderr + done.stdout, "complete saved runs must pass the expected-family check silently"
    for name in REGISTERED:
        assert (out / name).read_bytes() == (ref / name).read_bytes(), f"{name} differs from {reference}"


@pytest.mark.skipif(os.environ.get("FAIRSCREEN_SKIP_REGRESSION") == "1", reason="FAIRSCREEN_SKIP_REGRESSION=1")
def test_registered_kaggle_numbers_are_unchanged(tmp_path):
    rerun_and_compare(tmp_path, "--records", "results/records", "results/final_kaggle")


@pytest.mark.skipif(os.environ.get("FAIRSCREEN_SKIP_REGRESSION") == "1", reason="FAIRSCREEN_SKIP_REGRESSION=1")
def test_registered_cluster_numbers_are_unchanged(tmp_path):
    rerun_and_compare(tmp_path, "--local", "results/cluster_records", "results/final_cluster")
