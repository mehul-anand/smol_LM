# Change Log

- what are we changing from ./bigram.py to ./bigram_attention.py

## Changes :

- line 77 :
  - `def __init__(self, vocab_size):` : `def __init__(self):`
    - we already have `vocab_size` defined

- line 113 :
  - `model = BigramLanguageModel(vocab_size)` : `model = BigramLanguageModel()`
  - same logic

- line 80 :
  - `self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)` : `self.token_embedding_table = nn.Embedding(vocab_size, n_embed)`
    - idea is to introduce some more phases insteda of going for the logitsd directly
    - `n_embed` : number of embedding dimensions
    - `n_embed = 32` : pretty decent value
    - so now :
    - `self.token_embedding_table = nn.Embedding(vocab_size, n_embed)` : token embeddings
    - this to `logits` would be done via some `Linear` layers
      - `self.lm_head = nn.Linear(n_embed, vocab_size)`

- line 82 :
  - `self.lm_head = nn.Linear(n_embed, vocab_size)` :
    - now we are using `Linear` as an intermediary level

- line 80 - 82 regiion :

```python
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
```

to

```python
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
```

- so each vector from `0` to `block_size-1` will get a vector

- line 87 to 93

```python
        # pos_embedding part
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # pos_embedding part
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        new_emb = (
            tok_emb + pos_emb
        )  # (B,T,C) [a new dimension of B is added to make the addition make sense and it is broadcasted back]
        logits = self.token_embedding_table(new_emb)  # (B,T,vocab_size)
```

- we take out the `T` using the `idx.shape` and then use that to have integers from `0` to `T-1` using `torch.arange`
- they (integers) get embedded through the table to create a tensor of (T,C)
- this `new_emb` sotres the token identites along with their positions in which they occur
  > for the current setup of the bigram model : this method/approach is not that useful because in whichever position you are in, it is translation invariant but it will help as we develop more complex self attention
