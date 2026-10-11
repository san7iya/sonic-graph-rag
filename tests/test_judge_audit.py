from scripts.judge_audit import agreement, sample_claims


def make_progress(n_supported, n_unsupported):
    claims = ([{"claim": f"s{i}", "supported": True} for i in range(n_supported)]
              + [{"claim": f"u{i}", "supported": False} for i in range(n_unsupported)])
    return {"q": {"claims": claims, "context": "ctx"}}


def test_sample_oversamples_unsupported_and_is_reproducible():
    progress = make_progress(40, 15)
    sample = sample_claims(progress, n=20)
    assert len(sample) == 20
    assert sum(not r["judge_supported"] for r in sample) == 10
    assert [r["claim"] for r in sample] == [r["claim"] for r in sample_claims(progress, n=20)]


def test_sample_fills_from_supported_when_few_unsupported():
    sample = sample_claims(make_progress(40, 3), n=20)
    assert len(sample) == 20 and sum(not r["judge_supported"] for r in sample) == 3


def test_agreement_counts_each_error_type():
    rows = [
        {"judge_supported": "True", "human_supported": "yes"},    # agree
        {"judge_supported": "False", "human_supported": "no"},    # agree
        {"judge_supported": "True", "human_supported": "no"},     # judge too lenient
        {"judge_supported": "False", "human_supported": "yes"},   # judge too strict
        {"judge_supported": "True", "human_supported": ""},       # unlabelled, ignored
    ]
    result = agreement(rows)
    assert result == {"labelled": 4, "agree": 2, "too_lenient": 1, "too_strict": 1, "rate": 0.5}
