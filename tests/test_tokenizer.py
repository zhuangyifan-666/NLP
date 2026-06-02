from nlp_llm.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    text = "hello tiny model"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == text
    assert tok.vocab_size == 260

