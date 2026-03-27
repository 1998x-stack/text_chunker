import pytest
from textchunker import utils as _utils

def test_count_tokens_returns_positive():
    assert _utils.count_tokens("hello world") > 0

def test_count_tokens_empty():
    assert _utils.count_tokens("") == 0

def test_whitespace_sentences_english():
    text = "First sentence. Second sentence. Third."
    sents = _utils.whitespace_sentences(text)
    assert len(sents) >= 2
    assert "First" in sents[0]

def test_whitespace_sentences_chinese():
    text = "第一句话。 第二句话！ 第三句话？"
    sents = _utils.whitespace_sentences(text)
    assert len(sents) >= 2

def test_whitespace_sentences_with_offsets():
    text = "First sentence. Second sentence. Third sentence."
    results = _utils.whitespace_sentences_with_offsets(text)
    for sent, start, end in results:
        assert text[start:end] == sent
        assert len(sent) > 0

def test_whitespace_sentences_with_offsets_no_gaps():
    text = "Hello world. How are you? I am fine."
    results = _utils.whitespace_sentences_with_offsets(text)
    for sent, start, end in results:
        assert start >= 0
        assert end <= len(text)
        assert start < end

def test_whitespace_sentences_with_offsets_single():
    text = "Just one sentence"
    results = _utils.whitespace_sentences_with_offsets(text)
    assert len(results) == 1
    assert results[0][0] == text.strip()

def test_clamp():
    assert _utils.clamp(5, 0, 10) == 5
    assert _utils.clamp(-1, 0, 10) == 0
    assert _utils.clamp(15, 0, 10) == 10

def test_sliding_windows():
    seq = ["a", "b", "c", "d"]
    result = list(_utils.sliding_windows(seq, 2))
    assert len(result) == 3
    assert result[0] == (0, ["a", "b"])
    assert result[2] == (2, ["c", "d"])

def test_sliding_windows_larger_than_seq():
    seq = ["a", "b"]
    result = list(_utils.sliding_windows(seq, 5))
    assert len(result) == 0
