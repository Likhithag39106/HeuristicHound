"""Link checker retry logic (no network)."""
from phase_a_data.check_links import check_one

LC = {"timeout_s": 1, "max_retries": 2, "backoff_s": 1}


def test_retries_rate_limit_then_succeeds():
    answers = iter([429, 503, 200])
    calls, sleeps = [], []
    st = check_one("u", LC, sleep=sleeps.append, status=lambda u, t, method: calls.append(method) or next(answers))
    assert st == 200 and len(calls) == 3 and sleeps == [1, 2]


def test_404_is_final_and_head_405_falls_back_to_get():
    assert check_one("u", LC, sleep=lambda s: None, status=lambda u, t, method: 404) == 404
    seen = []
    st = check_one("u", LC, sleep=lambda s: None,
                   status=lambda u, t, method: seen.append(method) or (405 if method == "HEAD" else 200))
    assert st == 200 and seen == ["HEAD", "GET"]


def test_gives_up_after_retries():
    st = check_one("u", LC, sleep=lambda s: None, status=lambda u, t, method: "URLError")
    assert st == "URLError"


def test_conftest_blocks_real_network():
    import urllib.request

    import pytest
    with pytest.raises(Exception, match="network access in a test"):
        urllib.request.urlopen("https://example.com", timeout=5)
