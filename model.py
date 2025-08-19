import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters -----
batch_size = 64
block_size = 256 # Context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------

torch.manual_seed(1337)

# Load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create character to index and index to character mappings
stoi = {ch:i for i,ch in enumerate(chars)} 
itos = {i:ch for i,ch in enumerate(chars)}

# Encode and decode functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode the text data
data = torch.tensor(encode(text), dtype=torch.long)

# Train-validation split
n = int(0.9*len(data)) # first 90% trian rest validation
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  data = train_data if split == 'train' else val_data
  # Choosing 4 random indexes from 0 to len-block_size to pick 4 blocks to make a batch
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device) , y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
       losses = torch.zeros(eval_iters)
       for k in range(eval_iters):
          X, Y = get_batch(split)
          logits, loss = model(X, Y)
          losses[k] = loss.item()
       out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False) # n_embd -> C
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Lower Triangular Matrix
    
    self.dropout = nn.Dropout(dropout) # Randomly turn offs some Neurons [Trains sub networks]

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)   # (B,T,C)
    q = self.query(x) # (B,T,C)  
    # Compute attention scores
    wei = q @ k.transpose(1,2) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0 , float('-inf') )
    wei = F.softmax(wei, dim = -1)
    wei = self.dropout(wei)
    # Perform weighted aggregation of values
    v = self.value(x)
    out = wei @ v
    return out

# Multi head attention -> multiple heads of self attention running in parallel
class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
    # Projection layer -> Dimension matching for residual connection
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate over channel dimension
    out = self.dropout(self.proj(out) )
    return out

class FeedForward(nn.Module):
  """ Just a MLP : A Simple layer followed by a non-linearity """
  
  def __init__(self, n_embd):  
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd), # 4 -> coming from paper
      nn.ReLU(),
      nn.Linear(4*n_embd, n_embd), # projection layer
      nn.Dropout(dropout)
    )
  def forward(self, x):
    return self.net(x)
  
# Building Block for Repetation (Nx)
class Block(nn.Module):
  """ Transformer block : communication followed by computation """
  def __init__(self, n_embd, n_head):
    # n_emb : embedding dimension, n_head : the number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head
    self.sa  = MultiHeadAttention(n_head, head_size) # Communication
    self.ffwd = FeedForward(n_embd) # Computation
    self.ln1 = nn.LayerNorm(n_embd) # Slight depart from paper LN added before multi head attention
    self.ln2 = nn.LayerNorm(n_embd)
  """ If reused same LayerNorm, forcing one set of parameters (gamma, β) to normalize both distributions. 
  That makes learning harder and usually hurts performance. """
  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # Residual connection
    x = x + self.ffwd(self.ln2(x))
    return x

# Super simple bigram model
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # We are embedding the properties of the characters into vectors of size n_embd
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # We are embedding the position of the characters in the sequence
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # The final layer to map the embedded vectors back to the vocabulary size
    self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
    self.ln_f =   nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensors of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C=n_embd)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C=n_embd)
    x = tok_emb + pos_emb # (B,T,C=n_embd)
    
    """ x = self.sa_head(x) # apply one head of self attention (B,T,C)
    x = self.ffwd(x) # After gathering all relevant data by self attention ff will think together on this """

    x = self.blocks(x)
    logits = self.lm_head(x) #(B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # -> -1 also works
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B,T) array of indices in this context
    for _ in range(max_new_tokens):
      # (idx can't run out of block_size due to position embedding table)
      # Crop the idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits , loss = self(idx_cond)
      # Focus only in the last time step
      logits = logits[:,-1,:] # becomes (B, C)
      # Apply softmax to get probabilities
      probs = F.softmax(logits, dim = -1) # (B,C)
      # Sample from the distribution
      idx_next = torch.multinomial(probs, num_samples= 1) # (B,1)
      # Append sampled index to running sequence
      idx = torch.cat((idx, idx_next), dim= 1) # (B, T+1)
    return idx

model = BigramLanguageModel()
m = model.to(device)

# Create Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Every once in a while evaluate loss on train and val sets
    if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model 
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))
print(device)
