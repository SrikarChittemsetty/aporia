from api.claims import CLAIM_MARKERS, resolve_claim


def test_builtin_topics_resolve():
    claim, was_topic, err = resolve_claim("free will")
    assert claim == "Humans have free will" and was_topic and err is None
    claim, was_topic, _ = resolve_claim("Existence of God?")
    assert claim == "God exists" and was_topic


def test_claims_pass_through_unchanged():
    for q in ["free will is an illusion", "determinism makes morality meaningless",
              "God exists", "morality requires religion"]:
        claim, was_topic, err = resolve_claim(q)
        assert claim == q and not was_topic and err is None


def test_claim_markers_catch_predicates():
    assert CLAIM_MARKERS.search("beauty is truth")
    assert CLAIM_MARKERS.search("we should abolish suffering")
    assert not CLAIM_MARKERS.search("personal identity")
