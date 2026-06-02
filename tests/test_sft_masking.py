import pytest

torch = pytest.importorskip("torch")

from nlp_llm.data_sft import build_sft_features
from nlp_llm.tokenizer import ByteTokenizer


def test_response_only_mask_has_prompt_ignored_and_response_trained():
    tok = ByteTokenizer()
    item = build_sft_features(tok, "Say hi.", "Hello.", max_length=128)
    labels = item["labels"]
    mask = item["loss_mask"]
    assert labels.shape == mask.shape
    ignored = labels == -100
    trained = labels != -100
    assert ignored.any()
    assert trained.any()
    assert torch.all(mask[ignored] == 0)
    assert torch.all(mask[trained] == 1)

