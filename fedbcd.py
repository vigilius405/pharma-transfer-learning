#FedBCD -- https://arxiv.org/pdf/1912.11187

"""
Sequential FedBCD (two-party vertical FL) - PyTorch simulation

Assumptions:
- embA: numpy array shape (N, dimA)  -- features at party A (also holds labels)
- embB: numpy array shape (N, dimB)  -- features at party B
- labels: numpy array shape (N,)     -- 0/1 labels held by party A

The code simulates two parties in one process. In practice, parties would
run on separate machines/processes and only exchange H_k (intermediate scalars).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score #roc_auc_score, 
#from tqdm import trange
from omniscient import LABEL, CLINTOX_CATS, TOX21_CATS

# -------------------------
# Simple dataset wrapper
# -------------------------
class VerticalEmbeddingDataset(Dataset):

    def __init__(self, A: pd.DataFrame, B: pd.DataFrame, A_embed: np.ndarray, B_embed: np.ndarray,
                A_class: str=LABEL, A_cats: list[str]=CLINTOX_CATS, B_cats: list[str]=TOX21_CATS,
                gets_embeddings: list[int]=[1,1]):
        #tracking indices of merge so can reshape embeddings in right way
        A = A.reset_index(drop=True).copy()
        B = B.reset_index(drop=True).copy()

        A['_row_idx_A'] = A.index
        B['_row_idx_B'] = B.index

        #merging into shared indices
        merged = pd.merge(A, B, on='smiles_can', how='inner')
        #merged_embeddings = merged[A_cats + B_cats]

        #pruning embeddings to match dataframe -- these should be the same
        A_embed_pruned = A_embed[merged['_row_idx_A'].values]
        B_embed_pruned = B_embed[merged['_row_idx_B'].values]
        np.testing.assert_allclose(A_embed_pruned, B_embed_pruned), 'A and B embeddings should be same at this stage'

        #converting dataframe to np.ndarray
        a_cats_numpy = merged[A_cats].to_numpy()
        b_cats_numpy = merged[B_cats].to_numpy()
        if gets_embeddings[0]:
          self.embA = torch.tensor(np.hstack((a_cats_numpy, A_embed_pruned))) #dim2 = 385
        else:
          self.embA = torch.tensor(a_cats_numpy) #dim2 = 1
        if gets_embeddings[1]:
          self.embB = torch.tensor(np.hstack((b_cats_numpy, A_embed_pruned))) #dim2 = 396
        else:
          self.embB = torch.tensor(b_cats_numpy) #dim2 = 12
        self.labels = torch.tensor(merged[A_class].to_numpy()).unsqueeze(1) # shape (n, 1)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #print(self.embeddings[idx].to(torch.float32).dtype, self.labels[idx].dtype)
        return self.embA[idx].to(torch.float32), self.embB[idx].to(torch.float32), self.labels[idx].to(torch.float32)

# -------------------------
# Party model blocks (small MLP returning scalar logits)
# -------------------------
class PartyBlock(nn.Module): #same as MLPClassifier
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

# -------------------------
# FedBCD training (sequential) <- TODO: use parallel! if can
# -------------------------
def fedbcd_sequential_train(
    df_reordered_a, df_reordered_b, embA, embB, gets_embeddings,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    q_local=1,               # number of local updates per party per batch
    hidden_dim=128,
    device=None,
    val_split=0.2
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    #dataset = VerticalEmbeddingDataset(embA, embB, labels)
    dataset = VerticalEmbeddingDataset(df_reordered_a, df_reordered_b, embA, embB, gets_embeddings=gets_embeddings)
    n = len(dataset)
    # train/val split indices
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(n * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    # initialize party blocks
    dimA = dataset.embA.shape[1]
    dimB = dataset.embB.shape[1]
    partyA = PartyBlock(dimA, hidden_dim=hidden_dim).to(device)  # holds labels too
    partyB = PartyBlock(dimB, hidden_dim=hidden_dim).to(device)

    # optimizers per party (they only update their own params)
    optA = torch.optim.Adam(partyA.parameters(), lr=lr)
    optB = torch.optim.Adam(partyB.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    for epoch in range(1, epochs+1):
        partyA.train()
        partyB.train()
        epoch_losses = []

        # Sequential FedBCD: iterate minibatches, then for each party sequentially do Q local updates
        for batch in train_loader:
            xA_batch, xB_batch, y_batch = batch
            xA_batch = xA_batch.to(device)
            xB_batch = xB_batch.to(device)
            y_batch = y_batch.to(device)

            # Step 0: parties compute their current contributions H_k
            # (In real deployment they would exchange H_k across network)
            with torch.no_grad():
                H_A = partyA(xA_batch)  # (batch,1)
                H_B = partyB(xB_batch)  # (batch,1)

            # We perform sequential updates (party A then party B), each does Q local updates
            # Party A updates θ_A while treating H_B fixed (detached)
            for j in range(q_local):
                # recompute H_A with gradients enabled (since θ_A may have changed)
                optA.zero_grad()
                H_A_train = partyA(xA_batch)                 # depends on θ_A
                H_B_fixed = H_B.detach()                     # fixed contribution from party B
                logit = H_A_train + H_B_fixed                # combined logit
                loss = criterion(logit, y_batch)
                loss.backward()
                optA.step()

                # update H_A for next local iteration (so next iteration uses new θ_A)
                with torch.no_grad():
                    H_A = partyA(xA_batch)

            # After party A finished Q updates, exchange (in simulation we just have H_A)
            # Party B updates θ_B while treating H_A fixed
            for j in range(q_local):
                optB.zero_grad()
                H_B_train = partyB(xB_batch)
                H_A_fixed = H_A.detach()
                logit = H_A_fixed + H_B_train
                loss_b = criterion(logit, y_batch)   # party B uses labels only to compute gradient here in sim
                loss_b.backward()
                optB.step()

                with torch.no_grad():
                    H_B = partyB(xB_batch)

            # record loss for monitoring using final H_A & H_B
            with torch.no_grad():
                final_logit = H_A + H_B
                final_loss = criterion(final_logit, y_batch).item()
                epoch_losses.append(final_loss)

        # --- validation after epoch ---
        partyA.eval()
        partyB.eval()
        ys = []
        preds = []
        losses_v = []
        with torch.no_grad():
            for xA_v, xB_v, y_v in val_loader:
                xA_v = xA_v.to(device)
                xB_v = xB_v.to(device)
                y_v = y_v.cpu().numpy().ravel()
                HAv = partyA(xA_v)
                HBv = partyB(xB_v)
                logits_v = (HAv + HBv).cpu().numpy().ravel()
                #print(type(logits_v), type(y_v))
                loss_v = criterion(torch.from_numpy(logits_v), torch.from_numpy(y_v)).item()
                probs = 1.0 / (1.0 + np.exp(-logits_v))
                ys.append(y_v)
                preds.append(probs)
                losses_v.append(loss_v)

        ys = np.concatenate(ys)
        preds = np.concatenate(preds)
        #losses_v = np.concatenate(losses_v)
        # try:
        #     auc = roc_auc_score(ys, preds)
        # except ValueError:
        #     auc = float('nan')  # when only one class present on val
        acc = accuracy_score(ys, (preds > 0.5).astype(int))

        print(f"Epoch {epoch:02d} | train_loss={np.mean(epoch_losses):.4f} | val_loss={sum(losses_v)/len(losses_v):.4f} | val_acc={acc:.4f}")

    return partyA, partyB

