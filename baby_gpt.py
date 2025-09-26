# This is a BabyGPT Project by Arsenic (6/9/2025). 
# The main purpose is to understand how LLMs/AIs (in specific, GPTs) work under the hood.


# We are importing PyTorch, which is a large, open source library for machine learning
import torch

# We're simply calling the torch module, "torch.nn", as "nn", to make the code simpler and easier to call
import torch.nn as nn

# a torch module WITHIN nn, which is called for a full, control for functions
import torch.nn.functional as F


# Basic character-level tokenizer section :D

# Data for training.
data = "hello world, this is a longer data for training."

# In this code, data is parsed within the chars variable; removing any duplicates and organize it
chars = sorted(list(set(data)))

# The beginning of vectorization, this is where the strings got assigned to numbers
stoi = {ch: i for i, ch in enumerate(chars)}

# Inverting the "stoi" line, this changes from the numbers to strings again
itos = {i: ch for ch, i in stoi.items()}

# Definition 
def encode(s):
    return [stoi[c] for c in s]

#
def decode(l):
    return "".join([itos[i] for i in l])


# Mini transformer block

#
class TinySelfAttention(nn.Module):

    #
    def __init__(self, embed_size):
        super().__init__()
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear (embed_size, embed_size, bias=False)
        self.proj = nn.Linear(embed_size, embed_size)

    #
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Attention scores
        weights = q @ k.transpose(-2, -1) / (C ** 0.5)
        weights = F.softmax(weights, dim=-1)
        out = weights @ v
        return self.proj(out)

# Transformer Block

#
class TinyTransformerBlock(nn.Module):

    #
    def __init__(self, embed_size):

        #
        super().__init__()

        #
        self.attn = TinySelfAttention(embed_size)

        #
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(), 
            nn.Linear(4 * embed_size, embed_size)
        )

        #
        self.norm1 = nn.LayerNorm(embed_size)

        #
        self.norm2 = nn.LayerNorm(embed_size)
    
    #
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# Full model

#
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, context_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(122880, embed_size)
        self.block = TinyTransformerBlock(embed_size)
        self.ln = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # shape (1, T)
        pos_emb = self.pos_embed(pos)  # shape (1, T, C)
        x = tok_emb + pos_emb
        x = self.block(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits
    
# Sampling

def sample(model, start, steps):
    model.eval()
    context = torch.tensor(encode(start), dtype=torch.long)[None, :]  
    for _ in range(steps):
        logits = model(context)
        logits = logits[:, -1, :]  
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).detach()  
        context = torch.cat([context, next_token], dim=1)  
    return decode(context[0].tolist())

# Training it

# Dummy Dataset

vocab_size = len(stoi)
context_size = 16
embed_size = 32

model = TinyGPT(vocab_size, embed_size, context_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)

# Training loop (very basic)
for step in range(6500):
    idx = torch.tensor([encode(data[:context_size])], dtype=torch.long)
    targets = torch.tensor(encode(data[1:context_size + 1]), dtype=torch.long)
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step{step}, Loss:{loss.item():.4f}")

# Trying it out
print(sample(model, start="h", steps=20))