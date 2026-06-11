import pytest

torch = pytest.importorskip("torch")

from advanced.lora import LoRAConfig, inject_lora, mark_only_lora_trainable, merge_lora_weights
from nlp_llm.model import GPT, GPTConfig, load_gpt_from_checkpoint, save_checkpoint


def test_lora_injection_forward_and_merge():
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0))
    replaced = inject_lora(model, LoRAConfig(rank=2, alpha=4.0, target_modules=("q_proj", "v_proj")))
    assert replaced
    mark_only_lora_trainable(model)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    assert torch.isfinite(model(x, labels=y)["loss"])
    merged = merge_lora_weights(model)
    assert merged == replaced
    assert torch.isfinite(model(x, labels=y)["loss"])


def test_lora_checkpoint_roundtrip(tmp_path):
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0))
    lora = LoRAConfig(rank=2, alpha=4.0, target_modules=("q_proj",))
    inject_lora(model, lora)
    ckpt = tmp_path / "lora.pt"
    save_checkpoint(
        ckpt,
        model,
        extra={
            "tokenizer_path": "unused",
            "lora": {"enabled": True, **lora.to_dict()},
        },
    )
    loaded = load_gpt_from_checkpoint(ckpt)
    x = torch.randint(0, 64, (2, 8))
    assert loaded(x)["logits"].shape == (2, 8, 64)
