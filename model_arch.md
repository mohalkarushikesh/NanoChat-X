Here’s a Mermaid diagram showing only the NanoChat-X GPT model architecture, from input tokens through causal self-attention and transformer blocks to next-token prediction.

```mermaid
flowchart TD
    A["Input Token IDs<br/>[B, T]"] --> B["Token Embedding<br/>wte"]
    A --> C["Learned Positional Embedding<br/>wpe"]
    B --> D["Add Embeddings"]
    C --> D
    D --> E["Embedding Dropout"]

    E --> F

    subgraph TR["Transformer Blocks × 8"]
        F["Input x"]
        F --> G["LayerNorm 1"]
        G --> H["Causal Multi-Head Self-Attention<br/>8 Heads"]
        H --> I["Residual Add: x + Attention"]
        F --> I

        I --> J["LayerNorm 2"]
        J --> K["MLP: Linear 512 → 2048"]
        K --> L["GELU Activation"]
        L --> M["Linear 2048 → 512"]
        M --> N["Dropout"]
        N --> O["Residual Add: x + MLP"]
        I --> O
    end

    O --> P["Final LayerNorm"]
    P --> Q["LM Head<br/>512 → Vocabulary Size 256"]
    Q --> R["Next-Token Logits<br/>[B, T, V]"]

    B -. "Shared weights" .-> Q

    style TR fill:#eaf2ff,stroke:#4472c4,stroke-width:2px
    style H fill:#d9ead3,stroke:#38761d
    style Q fill:#fce5cd,stroke:#e69138
    style R fill:#fff2cc,stroke:#bf9000
```

## Architecture at a glance

- Input embeddings: Token embeddings + learned positional embeddings.
- Transformer: 8 pre-LayerNorm blocks, each with 8-head causal self-attention and a 4× feed-forward MLP.
- Causal masking: Each token can attend only to itself and previous tokens.
- Output: Final LayerNorm → LM head → next-token logits.
- Weight tying: The LM head shares its weights with the token embedding.
- Configuration: 512 embedding dimensions, 256-token context window, vocabulary size 256.

Note: The diagram shows one transformer block in detail, repeated 8 times. The actual model predicts the next token during training by comparing its logits with the shifted target tokens.