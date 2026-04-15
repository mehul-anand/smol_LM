import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
# check wether GPU is available on the device or not
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)  # mps for mac
# for evaluation iterations
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "../../data/input.txt")
output_path = os.path.join(script_dir, "../outputs/smol_v1.txt")
device_str = f"Using device: {device}\n"

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# for google collab
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# basic tokenizer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Indicating to PyTorch that we don't need .backward() here : saves computation
@torch.no_grad()  # context manager
# avergaing up the losses over multiple batches
def estimate_loss():
    out = {}
    # setting the model to evaluation phase
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # setting the model to training phase
    model.train()
    return out


# implementing self-attention block
class Head(nn.Module):
    tril: torch.Tensor

    # one head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # tril is not a parameter of the PyTorch module and so we use a register_buffer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute affinities/attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # for masking the future values (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    # multiple attention heads of self-attention in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        # running them in parallel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concatinating the outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # dim=-1 -> channel (C) dimension
        # we are now doing a linear transformation over the self attention we got : this is for the residual connections part
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # dimensionality from 512 to 2048
            nn.ReLU(),
            # the projection layer
            nn.Linear(
                4 * n_embed, n_embed
            ),  # dimensionality while making projections back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # this is the transformer block
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        # implementing LayerNorm
        self.ln1 = nn.LayerNorm(
            n_embed
        )  # size of the LayerNorm is 32 (will work per token)
        self.ln2 = nn.LayerNorm(n_embed)

    # with residual connections
    # fork off do some communication and then comeback

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # pos_embedding part
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # blocks
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # pos_embedding part
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # pos_embedding part
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        new_emb = tok_emb + pos_emb  # (B,T,C)
        new_emb = self.blocks(new_emb)  # (B,T,C)
        new_emb = self.ln_f(new_emb)  # (B,T,C)
        logits = self.lm_head(new_emb)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the last block_size tokens so that it does not run out of scope
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

with open(output_path, "w") as f:
    f.write(device_str)
    print(device_str, end="")
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            line = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            print(line)
            f.write(line + "\n")
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    print(generated)
    f.write("\n--- Generated Output ---\n")
    f.write(generated + "\n")
