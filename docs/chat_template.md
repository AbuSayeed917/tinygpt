# Input format and chat template

The special tokens are single ids in the vocabulary (`bpe/tokenizer.py`). The
tokenizer's plain `encode()` never produces them, and only
`encode_with_special()` turns their strings into special ids. That method is used
by the template builders in this repo, never on user input.

## Pretraining (Phase 4)

Documents are joined into one long stream, each followed by `<|endoftext|>`, and
cut into fixed-length windows. The end marker teaches the model where one
document stops, so it doesn't carry facts from one page into the next.

```
<doc 1 tokens> <|endoftext|> <doc 2 tokens> <|endoftext|> ...
```

## Chat (Phase 5 fine-tuning and Phase 7 serving)

```
<|system|>{system prompt}<|end|>
<|user|>{message}<|end|>
<|assistant|>{reply}<|end|>
<|user|>{message}<|end|>
<|assistant|>
```

- At inference, the prompt ends with `<|assistant|>` and generation stops at
  `<|end|>` or at the token limit.
- **Loss masking:** during fine-tuning, loss is computed only on assistant reply
  tokens and their closing `<|end|>`. The model learns to write replies, not to
  imitate users or recite the system prompt.
- **Grounded answers:** retrieved knowledge goes inside the system turn, fenced as
  data:

  ```
  <|system|>You are the assistant on Sayeed's site. Answer only from the facts below.
  Facts:
  """
  {retrieved chunk 1}
  {retrieved chunk 2}
  """<|end|>
  ```

## Why the roles can't be spoofed

| Attack | What happens |
|---|---|
| User types `<|assistant|>` or `<|system|>` | `encode()` turns it into ordinary character tokens, which are never the role ids the model was trained on |
| User types "Ignore previous instructions" | No structural effect. Handled by refusal examples in fine-tuning and by the red-team eval (Phase 6) |
| Retrieved text contains instructions | Fenced inside `"""` as data, and the model is trained on examples that ignore such instructions |

The server's input validation (Phase 7) also rejects messages containing the
special-token strings, so they are blocked at two layers.
