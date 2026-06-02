import pytest

torch = pytest.importorskip("torch")

from nlp_llm.model import GPT, GPTConfig


def test_model_forward_loss_and_generate():
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0))
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    out = model(x, labels=y)
    assert out["logits"].shape == (2, 8, 64)
    assert torch.isfinite(out["loss"])
    gen = model.generate(x[:, :3], max_new_tokens=4, top_k=10)
    assert gen.shape == (2, 7)

