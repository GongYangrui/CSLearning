import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 64
block_size = 256
max_iter = 3000
eval_interval = 300
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200
n_embedding = 32

# 读取数据
with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for index, char in enumerate(chars)}
encode = lambda sentence: [stoi[char] for char in sentence]
decode = lambda index_list: "".join([itos[index] for index in index_list])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)

    def forward(self, x): # x (batch_size, block_size, embedding_size)
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x) # x (batch_size, block_size, head_size)
        wei = q @ k.transpose(-2, -1)
        tril = torch.tril(torch.ones(T, T))
        wei = wei.masked_fill(tril==0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (batch_size, block_size, block_size)
        out = wei @ v # (batch_size, block_size, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embedding, n_embedding)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding) # 位置编码，每个位置的编码不同
        # self.sa_heads = MultiHeadAttention(4, n_embedding // 4)
        self.lm_head = nn.Linear(n_embedding, vocab_size)
        # self.ffwd = FeedForward(n_embedding)
        self.block = nn.Sequential(
            Block(n_embedding, 4),
            Block(n_embedding, 4),
            Block(n_embedding, 4),
            nn.LayerNorm(n_embedding),
        )

    def forward(self, idx, target=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (batch_size, block_size, embedding_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (block_size, embedding_size)

        x = tok_emb + pos_emb
        # x = self.sa_heads(x) # (batch_size, block_size, embedding_size)
        # x = self.ffwd(x)
        x = self.block(x)
        logits = self.lm_head(x) # (batch_size, block_size, vocab_size)

        if target == None:
            loss = None
        else:
            batch, block, vocab = logits.shape
            logits = logits.view(batch * block, vocab)
            target = target.view(batch * block)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # (Batch_size, Block_size, Vocab_size)
            logits = logits[:, -1, :] # (Batch_size, Vocab_size)
            probs = F.softmax(logits, dim=1) # (Batch_size, Vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch_size, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (Batch_size, Block_size + 1)
        return idx

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

m = BigramLanguageModel()
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for epoch in range(100):
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())

idx = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))