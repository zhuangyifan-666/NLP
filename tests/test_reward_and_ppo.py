import pytest

torch = pytest.importorskip("torch")

from nlp_llm.model import GPTConfig
from nlp_llm.ppo import PPOConfig, PolicyWithValue, compute_ppo_loss, logprobs_from_logits
from nlp_llm.reward_model import RewardModel


def test_reward_pairwise_loss_finite():
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    model = RewardModel(cfg)
    chosen = torch.randint(0, 64, (2, 8))
    rejected = torch.randint(0, 64, (2, 8))
    mask = torch.ones_like(chosen)
    out = model.pairwise_loss(chosen, mask, rejected, mask)
    assert torch.isfinite(out["loss"])
    assert 0.0 <= float(out["accuracy"]) <= 1.0


def test_ppo_ministep_finite():
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    policy = PolicyWithValue(cfg)
    x = torch.randint(0, 64, (2, 8))
    actions = torch.randint(0, 64, (2, 8))
    out = policy(x)
    with torch.no_grad():
        old = logprobs_from_logits(out["logits"], actions)
        ref = old.clone()
    rewards = torch.tensor([0.5, 0.2])
    mask = torch.ones_like(actions, dtype=torch.float32)
    loss = compute_ppo_loss(out["logits"], out["values"], actions, old, ref, rewards, mask, PPOConfig())
    assert torch.isfinite(loss["loss"])

