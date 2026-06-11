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
    with torch.no_grad():
        old_out = policy(x)
        ref_out = policy(x)
        old = logprobs_from_logits(old_out["logits"], actions).detach()
        ref = logprobs_from_logits(ref_out["logits"], actions).detach()
        rewards = torch.tensor([0.5, 0.2]).detach()
    assert not old.requires_grad
    assert not ref.requires_grad
    assert not rewards.requires_grad
    out = policy(x)
    mask = torch.ones_like(actions, dtype=torch.float32)
    loss = compute_ppo_loss(out["logits"], out["values"], actions, old, ref, rewards, mask, PPOConfig())
    loss["loss"].backward()
    assert torch.isfinite(loss["loss"])
    assert any(p.grad is not None for p in policy.policy.parameters() if p.requires_grad)
    assert any(p.grad is not None for p in policy.value_head.parameters() if p.requires_grad)


def test_ppo_old_ref_reward_detached_for_reused_epochs():
    cfg = GPTConfig(vocab_size=32, block_size=12, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    policy = PolicyWithValue(cfg)
    ref_policy = PolicyWithValue(cfg).eval()
    reward_model = RewardModel(cfg).eval()
    for p in ref_policy.parameters():
        p.requires_grad = False
    for p in reward_model.parameters():
        p.requires_grad = False
    full = torch.randint(1, 32, (2, 8))
    model_in = full[:, :-1]
    actions = full[:, 1:]
    mask = torch.ones_like(actions, dtype=torch.float32)
    with torch.no_grad():
        old_out = policy(model_in)
        ref_out = ref_policy(model_in)
        old = logprobs_from_logits(old_out["logits"], actions).detach()
        ref = logprobs_from_logits(ref_out["logits"], actions).detach()
        rewards = reward_model(full, attention_mask=torch.ones_like(full)).detach()
    for cached in (old, ref, rewards):
        assert not cached.requires_grad
    for _ in range(2):
        out = policy(model_in)
        loss = compute_ppo_loss(out["logits"], out["values"], actions, old, ref, rewards, mask, PPOConfig())
        assert torch.isfinite(loss["loss"])
        policy.zero_grad(set_to_none=True)
        loss["loss"].backward()
