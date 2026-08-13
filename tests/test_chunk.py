from ingest.chunk import chunk_work, is_heading


def test_headings_detected():
    assert is_heading("BOOK I.")
    assert is_heading("SECTION VIII")
    assert is_heading("PART II.")
    assert is_heading("INTRODUCTION AND ANALYSIS.")
    assert is_heading("XII.")


def test_dialogue_replies_are_not_headings():
    # Short capitalized replies in Plato's dialogues must not become
    # citation paths (regression: greedy roman-numeral branch).
    for line in ["I do.", "I see.", "I agree.", "I suppose not.", "Yes, he said."]:
        assert not is_heading(line), line


def test_prose_is_not_a_heading():
    assert not is_heading("the will is determined by motives")
    assert not is_heading(
        "It is universally allowed that nothing exists without a cause of its existence."
    )


def test_chunk_work_tracks_headings_and_merges():
    para = "This is a sentence about liberty and necessity. " * 8  # ~64 words
    text = f"PART I.\n\n{para}\n\n{para}\n\n{para}\n\nPART II.\n\n{para}"
    chunks = list(chunk_work(text))
    assert all(c["citation"] in ("PART I.", "PART II.") for c in chunks)
    assert any(c["citation"] == "PART II." for c in chunks)
    # Merged chunks respect the max size.
    assert all(len(c["text"].split()) <= 320 for c in chunks)
