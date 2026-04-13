import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
# check wether GPU is available on the device or not
device = "cuda" if torch.cuda.is_available() else "cpu"
# for evaluation iterations
eval_iters = 200
n_embed = 32
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("test_notebooks/data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
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
    # even though this is not helping for this use case but this is considered as a good practice to know which mode the neural network is in
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
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # pos_embedding part
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # pos_embedding part
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # pos_embedding part
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        new_emb = (
            tok_emb + pos_emb
        )  # (B,T,C) [a new dimension of B is added to make the addition make sense and it is broadcasted back]
        new_emb = self.sa_head(new_emb)
        logits = self.lm_head(new_emb)  # (B,T,vocab_size)
        # bug: <fix> use self.lm_head instead of self.token_embedding_table to project to vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            # self.lm_head = nn.Linear(n_embed, vocab_size)
            # bug: <fix> remove this line - lm_head is already initialized in __init__, recreating it here creates duplicate parameters
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the last block_size tokens so that it does not run out of scope
            idx_cond = idx[:, -block_size:]
            # bug: <fix> should be idx[:, -block_size:] to crop last block_size tokens, not idx[:, block_size] which indexes a single column
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

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
