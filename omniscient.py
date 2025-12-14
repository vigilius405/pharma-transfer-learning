#One machine learning
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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

#simple classifier from embedding to classes
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

#single device training
#TODO: make sure can handle float (tox21), NaN (toxcast)
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
            #print((preds == y).sum().item(), y.size(0))
            correct += (preds == y).sum().item() #/ y.size(0)

    #print(len(dataloader.dataset))
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)
