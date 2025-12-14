#Horizontal federated learning

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import argparse
import pickle
import pandas as pd
import copy

# ----------------------
# Dataset Definition
# ----------------------

CLINTOX_CATS = ['CT_TOX'] #[]
TOX21_CATS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
              'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
LABEL = 'FDA_APPROVED'

class EmbeddingDataset(Dataset):

    def __init__(self, A: pd.DataFrame, B: pd.DataFrame, A_embed: np.ndarray, B_embed: np.ndarray,
                A_class: str=LABEL, A_cats: list[str]=CLINTOX_CATS, B_cats: list[str]=TOX21_CATS):

        if B is None:
          self.embeddings = torch.tensor(np.hstack((A[A_cats].to_numpy(), A_embed))) # shape (n, 1 + 384)
          self.labels = torch.tensor(A[A_class].to_numpy()).unsqueeze(1) #shape (n, 1)

        else:
          #tracking indices of merge so can reshape embeddings in right way
          A = A.reset_index(drop=True).copy()
          B = B.reset_index(drop=True).copy()

          A['_row_idx_A'] = A.index
          B['_row_idx_B'] = B.index

          #merging into shared indices
          merged = pd.merge(A, B, on='smiles_can', how='inner')
          merged_embeddings = merged[A_cats + B_cats]

          #pruning embeddings to match dataframe -- these should be the same
          A_embed_pruned = A_embed[merged['_row_idx_A'].values]
          B_embed_pruned = B_embed[merged['_row_idx_B'].values]
          np.testing.assert_allclose(A_embed_pruned, B_embed_pruned), 'A and B embeddings should be same at this stage'

          #converting dataframe to np.ndarray
          cats_merged = merged[A_cats + B_cats].to_numpy()
          self.embeddings = torch.tensor(np.hstack((cats_merged, A_embed_pruned))) # shape (n, 1 + 12 + 384)
          self.labels = torch.tensor(merged[A_class].to_numpy()).unsqueeze(1) # shape (n, 1)


    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        #print(self.embeddings[idx].to(torch.float32).dtype, self.labels[idx].dtype)
        return self.embeddings[idx].to(torch.float32), self.labels[idx].to(torch.float32)

# ----------------------
# Model Definition
# ----------------------

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, class_dim=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, class_dim)  # output logit
        )

    def forward(self, x):
        return self.net(x)

# ----------------------
# Utility Functions
# ----------------------
def reduce_tensor(x, world_size):
    """All-reduce a scalar/tensor value and return averaged result on all ranks."""
    if world_size <= 1:
        return x
    with torch.no_grad():
        t = x.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
    return t

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, world_size):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for xb, yb in dataloader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)
        #print(type(xb), xb.dtype, xb[0].dtype)
        #print(type(yb), yb.dtype, xb[0].dtype)

        optimizer.zero_grad()
        logits = model(xb)
        #print(type(logits), logits.dtype, logits[0].dtype)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.detach() * bs
        total_samples += bs

    # Synchronize before reduction
    if world_size > 1:
        dist.barrier()
    
    # Average loss across workers
    total_loss = reduce_tensor(total_loss.detach(), world_size)
    total_samples_tensor = torch.tensor(total_samples, dtype=torch.float32, device=device)
    total_samples_tensor = reduce_tensor(total_samples_tensor, world_size)
    avg_loss = (total_loss / total_samples_tensor).item()
    return avg_loss

def evaluate(model, dataloader, criterion, device, world_size):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            logits = model(xb)
            loss = criterion(logits, yb)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            bs = xb.size(0)
            total_loss += loss * bs
            total_samples += bs
            correct += (preds == yb).sum().float()

    # Synchronize before reduction
    if world_size > 1:
        dist.barrier()
    
    # Reduce across workers
    total_loss = reduce_tensor(total_loss.detach(), world_size)
    total_samples_tensor = reduce_tensor(torch.tensor(total_samples, device=device, dtype=torch.float32), world_size)
    correct = reduce_tensor(correct.detach(), world_size)

    avg_loss = (total_loss / total_samples_tensor).item()
    accuracy = (correct / total_samples_tensor).item()
    return avg_loss, accuracy

# ----------------------
# Main Training Function
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--input_dim', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    # Get distributed training parameters from environment
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    print(f"[Rank {rank}] Starting worker {rank}/{world_size}", flush=True)
    
    # Initialize process group
    dist.init_process_group(backend='gloo')  # Use 'nccl' if you have GPUs
    
    # Set device
    device = torch.device("cpu")  # Change to f"cuda:{local_rank}" if using GPUs
    
    is_main = (rank == 0)
    
    # ----------------------
    # Load or Create Your Data Here
    # ----------------------
    
    if is_main:
        print("Loading data...", flush=True)
    
    #dataset = EmbeddingDataset(embeddings, labels)
    with open(args.dataset, 'rb') as inp:
        train_dataset = pickle.load(inp)
    
    val_dataset = copy.deepcopy(train_dataset) 

    n_train = int(len(train_dataset) * 0.8)
    n_val = len(train_dataset) - n_train
    train_idcs = np.random.choice(len(train_dataset), n_train, replace=False)
    val_idcs = np.setdiff1d(np.arange(len(train_dataset)), train_idcs)

    train_dataset.embeddings = train_dataset.embeddings[train_idcs]
    train_dataset.labels = train_dataset.labels[train_idcs]

    val_dataset.embeddings = val_dataset.embeddings[val_idcs]
    val_dataset.labels = val_dataset.labels[val_idcs]
    
    # Hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True,
        drop_last=True  # Important: ensures equal batch counts across workers
    )
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=0,
        pin_memory=False
    )
    
    # Validation loader (using same data for simplicity) #NOTE TODO
    val_sampler = DistributedSampler(
        val_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler, 
        num_workers=0,
        pin_memory=False
    )
    
    if is_main:
        print(f"Created dataloaders: {len(dataloader)} batches per worker", flush=True)
    
    # ----------------------
    # Create Model
    # ----------------------
    input_dim = args.input_dim #embeddings.shape[1]
    model = MLPClassifier(input_dim=input_dim, hidden_dim=256, dropout=0.1).to(device)
    
    # Wrap with DDP
    model = DDP(model)
    
    if is_main:
        print(f"Model created and wrapped with DDP", flush=True)
    
    # ----------------------
    # Optimizer and Loss
    # ----------------------
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # ----------------------
    # Training Loop
    # ----------------------
    best_val_acc = 0.0
    
    if is_main:
        print("Starting training...", flush=True)
    
    for epoch in range(1, epochs + 1):
        # Set epoch for sampler (ensures different shuffle each epoch)
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, world_size)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, world_size)
        
        # Print results (only on main process)
        if is_main:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model (only on main process)
        if is_main and val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc
            }
            torch.save(checkpoint, 'best_model.pt')
            print(f"Saved new best model with accuracy: {val_acc:.4f}")
    
    # ----------------------
    # Cleanup
    # ----------------------
    if is_main:
        print("Training complete!", flush=True)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()