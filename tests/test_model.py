import math

import pytest
import torch

from model.gpt import GPT, IGNORE_INDEX, KVCache, ModelConfig

TINY = ModelConfig(
    vocab_size=64, context_length=32, d_model=32, n_layers=2, n_heads=4, ffn_hidden=64
)


@pytest.fixture
def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(TINY).eval()


def test_default_config_has_about_17m_parameters():
    # Arrange
    config = ModelConfig()
    per_layer = 4 * config.d_model**2 + 3 * config.d_model * config.ffn_hidden + 2 * config.d_model
    expected = config.vocab_size * config.d_model + config.n_layers * per_layer + config.d_model

    # Act
    model = GPT(config)

    # Assert
    assert model.num_params() == expected
    assert 16_000_000 < expected < 18_000_000


def test_output_head_is_tied_to_embedding(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.token_embedding.weight


def test_initial_loss_is_close_to_uniform_guessing(tiny_model):
    # Arrange: a fresh model should spread probability evenly: loss ~ ln(vocab).
    # Targets must be independent of inputs: with tied embeddings, a token's own
    # embedding scores highest against itself, so targets=inputs looks "learned".
    idx = torch.randint(0, TINY.vocab_size, (8, TINY.context_length))
    targets = torch.randint(0, TINY.vocab_size, (8, TINY.context_length))

    # Act
    _, loss = tiny_model(idx, targets=targets)

    # Assert
    assert loss is not None
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.05


def test_future_tokens_cannot_change_past_logits(tiny_model):
    # Arrange
    idx = torch.randint(0, TINY.vocab_size, (1, 16))
    changed = idx.clone()
    changed[0, 10] = (idx[0, 10] + 1) % TINY.vocab_size

    # Act
    original_logits, _ = tiny_model(idx)
    changed_logits, _ = tiny_model(changed)

    # Assert: positions before 10 are identical, position 10 onward may differ
    assert torch.allclose(original_logits[0, :10], changed_logits[0, :10], atol=1e-6)
    assert not torch.allclose(original_logits[0, 10:], changed_logits[0, 10:])


def test_kv_cache_decoding_matches_full_forward(tiny_model):
    # Arrange
    idx = torch.randint(0, TINY.vocab_size, (2, 12))
    full_logits, _ = tiny_model(idx)
    cache = KVCache(TINY, batch_size=2, device=idx.device, dtype=torch.float32)

    # Act: feed a 5-token prompt, then the rest one token at a time
    prompt_logits, _ = tiny_model(idx[:, :5], cache=cache)
    step_logits = [tiny_model(idx[:, t : t + 1], cache=cache)[0] for t in range(5, 12)]
    cached_logits = torch.cat([prompt_logits, *step_logits], dim=1)

    # Assert
    assert torch.allclose(cached_logits, full_logits, atol=1e-5)


def test_greedy_generation_is_deterministic_and_fills_context(tiny_model):
    prompt = torch.randint(0, TINY.vocab_size, (1, 4))

    first = tiny_model.generate(prompt, max_new_tokens=28, temperature=0)
    second = tiny_model.generate(prompt, max_new_tokens=28, temperature=0)

    assert first.shape == (1, TINY.context_length)
    assert torch.equal(first, second)


def test_generation_stops_at_stop_token(tiny_model):
    prompt = torch.randint(0, TINY.vocab_size, (1, 4))
    greedy_next = tiny_model.generate(prompt, max_new_tokens=1, temperature=0)[0, -1].item()

    output = tiny_model.generate(prompt, max_new_tokens=10, temperature=0, stop_token=greedy_next)

    assert output.shape == (1, 5)


def test_model_can_memorise_one_batch():
    # Arrange: if a model cannot overfit 4 short sequences, something is wired wrong
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (4, 17))
    inputs, targets = idx[:, :-1], idx[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Act
    for _ in range(150):
        _, loss = model(inputs, targets=targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Assert
    assert loss.item() < 0.1


def test_ignore_index_targets_add_no_loss(tiny_model):
    idx = torch.randint(0, TINY.vocab_size, (1, 8))
    masked = idx.clone()
    masked[0, :4] = IGNORE_INDEX

    _, masked_loss = tiny_model(idx, targets=masked)
    logits, _ = tiny_model(idx)
    expected = torch.nn.functional.cross_entropy(logits[0, 4:], idx[0, 4:])

    assert torch.allclose(masked_loss, expected, atol=1e-6)


def test_rejects_sequences_longer_than_context(tiny_model):
    too_long = torch.zeros((1, TINY.context_length + 1), dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds context"):
        tiny_model(too_long)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"d_model": 30, "n_heads": 4}, "divisible"), ({"d_model": 12, "n_heads": 4}, "even")],
)
def test_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig(**{**TINY.__dict__, **overrides})
