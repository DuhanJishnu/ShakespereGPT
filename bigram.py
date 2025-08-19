import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters -----
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
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

# Super simple bigram model
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):

    # idx and targets are both (B,T) tensors of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C=n_embd)
    logits = self.lm_head(tok_emb) #(B,T,vocab_size)

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
      #  get the predictions
      logits , loss = self(idx)
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
