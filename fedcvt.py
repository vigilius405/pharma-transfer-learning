#FedCVT

#PREP DATA AND RUN

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset #, DataLoader
from sklearn.metrics import accuracy_score
from omniscient import LABEL, CLINTOX_CATS, TOX21_CATS

class UnalignedEmbeddingDataset(Dataset):

    def __init__(self, A: pd.DataFrame, B: pd.DataFrame, A_embed: np.ndarray, B_embed: np.ndarray,
                A_class: str=LABEL, A_cats: list[str]=CLINTOX_CATS, B_cats: list[str]=TOX21_CATS,
                train_val_split=0.8):

        #tracking indices of merge so can reshape embeddings in right way
        A = A.reset_index(drop=True).copy()
        B = B.reset_index(drop=True).copy()
        print(A.shape, B.shape, A_embed.shape, B_embed.shape)

        A['_row_idx_A'] = A.index
        B['_row_idx_B'] = B.index

        #merging into shared indices
        merged = pd.merge(A, B, on='smiles_can', how='inner')
        merged_embeddings = merged[A_cats + B_cats]

        #pruning embeddings to match dataframe -- these should be the same
        A_embed_pruned = A_embed[merged['_row_idx_A'].values]
        A_ul_idcs = np.setdiff1d(np.arange(A_embed.shape[0]), merged['_row_idx_A'].values)
        A_ul_embed = A_embed[A_ul_idcs]
        B_embed_pruned = B_embed[merged['_row_idx_B'].values]
        B_ul_idcs = np.setdiff1d(np.arange(B_embed.shape[0]), merged['_row_idx_B'].values)
        B_ul_embed = B_embed[B_ul_idcs]
        np.testing.assert_allclose(A_embed_pruned, B_embed_pruned), 'A and B embeddings should be same at this stage'

        #additional non-structural features
        A_additional_ul = (A.iloc[A_ul_idcs,:])[A_cats].to_numpy()
        B_additional_ul = (B.iloc[B_ul_idcs,:])[B_cats].to_numpy()
        A_additional_al = merged[A_cats].to_numpy()
        B_additional_al = merged[B_cats].to_numpy()

        #getting labels
        self.labels_ul = (A.iloc[A_ul_idcs,:])[A_class].to_numpy()
        self.labels_al = merged[A_class].to_numpy()
        self.labels = torch.tensor(np.concatenate((self.labels_al, self.labels_ul), axis=0))
        self.n_aligned = self.labels_al.shape[0]

        #putting together the pieces
        self.A_al = torch.tensor(np.hstack((A_embed_pruned, A_additional_al)))
        self.B_al = torch.tensor(np.hstack((B_embed_pruned, B_additional_al)))
        self.A_ul = torch.tensor(np.hstack((A_ul_embed, A_additional_ul)))
        self.B_ul = torch.tensor(np.hstack((B_ul_embed, B_additional_ul)))
        self.embA = torch.tensor(np.vstack((self.A_al, self.A_ul)))
        self.embB = torch.tensor(np.vstack((self.B_al, self.B_ul)))

        assert len(self.A_al) == len(self.B_al) and len(self.A_al) == len(self.labels_al) and \
               len(self.A_ul) == len(self.labels_ul), "Mismatch in alignment lengths!"

        #random division for training
        self.train_val_split = train_val_split
        self.al_train_idcs = np.random.choice(np.arange(self.n_aligned), int(self.train_val_split * self.n_aligned), replace=False).tolist()
        self.al_val_idcs = np.setdiff1d(np.arange(self.n_aligned), self.al_train_idcs).tolist()
        #print(np.random.choice(np.arange(self.A_ul), int(self.train_val_split * len(self.A_ul)), replace=False))
        self.A_ul_train_idcs = (np.random.choice(np.arange(len(self.A_ul)), int(self.train_val_split * len(self.A_ul)), replace=False) \
                                + self.n_aligned).tolist()
        self.B_ul_train_idcs = (np.random.choice(np.arange(len(self.B_ul)), int(self.train_val_split * len(self.B_ul)), replace=False) \
                                + self.n_aligned).tolist()
        self.A_ul_val_idcs = (np.setdiff1d(np.arange(len(self.A_ul)), self.A_ul_train_idcs) + self.n_aligned).tolist()
        self.B_ul_val_idcs = (np.setdiff1d(np.arange(len(self.B_ul)), self.B_ul_train_idcs) + self.n_aligned).tolist()

    def __len__(self):
        #len of aligned, A, B
        return self.n_aligned, len(self.embA), len(self.embB)

    def get_train(self):
        print(type(self.embA), type(self.al_train_idcs), type(self.A_ul_train_idcs))
        return self.embA[self.al_train_idcs + self.A_ul_train_idcs], \
               self.embB[self.al_train_idcs + self.B_ul_train_idcs], \
               self.labels[self.al_train_idcs + self.A_ul_train_idcs]

    def get_val(self, aligned_only: bool=True):
        if aligned_only:
            return self.embA[self.al_val_idcs], self.embB[self.al_val_idcs], \
                   self.labels[self.al_val_idcs]
        return self.embA[self.al_val_idcs + self.A_ul_val_idcs], \
               self.embB[self.al_val_idcs + self.B_ul_val_idcs], \
               self.labels[self.al_val_idcs + self.A_ul_val_idcs]

    # def __getitem__(self, idx):
    #     #print(self.embeddings[idx].to(torch.float32).dtype, self.labels[idx].dtype)
    #     return self.embeddings[idx].to(torch.float32), self.labels[idx].to(torch.float32)


# -------------------------
# Dataset
# -------------------------
# class FedCVTDataset(Dataset): #TODO EDIT THIS
#     def __init__(self, embeddings, labels=None):
#         """
#         embeddings: numpy array or tensor (N, feature_dim)
#         labels: numpy array or tensor (N,) or (N,1), None if no labels
#         """
#         if isinstance(embeddings, np.ndarray):
#             self.embeddings = torch.from_numpy(embeddings).float()
#         else:
#             self.embeddings = embeddings.float()
        
#         if labels is not None:
#             if isinstance(labels, np.ndarray):
#                 self.labels = torch.from_numpy(labels).float()
#             else:
#                 self.labels = labels.float()
            
#             if self.labels.dim() == 1:
#                 self.labels = self.labels.unsqueeze(1)
#         else:
#             self.labels = None
    
#     def __len__(self):
#         return len(self.embeddings)
    
#     def __getitem__(self, idx):
#         if self.labels is not None:
#             return self.embeddings[idx], self.labels[idx]
#         return self.embeddings[idx]

# -------------------------
# Neural Network Modules
# -------------------------
class RepresentationNet(nn.Module): #same as the prev modules
    """Neural network for learning representations"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class AttentionEstimator(nn.Module):
    """Scaled dot-product attention for representation estimation"""
    def __init__(self, T=0.5):
        super().__init__()
        self.T = T  # Sharpening temperature
    
    def sharpen(self, p):
        """Sharpening operation from the paper"""
        u = torch.pow(p, 1.0 / self.T)
        return u / torch.sum(u, dim=1, keepdim=True)
    
    def forward(self, Q, K, V):
        """
        Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        Q: query (m, d_k)
        K: key (n, d_k)
        V: value (n, d_v)
        Returns: (m, d_v)
        """
        d_k = K.shape[1]
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        attention = torch.softmax(scores, dim=-1)
        # Apply sharpening
        attention = self.sharpen(attention)
        # Weighted sum of values
        return torch.matmul(attention, V)

class Classifier(nn.Module):
    """Softmax classifier"""
    def __init__(self, input_dim, hidden_dim=128, num_classes=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# -------------------------
# FedCVT Training
# -------------------------
def fedcvt_train(
    A_data, B_data, A_labels,
    n_aligned,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    repr_dim=64,
    hidden_dim=128,
    t=0.5,  # Pseudo-label threshold
    T=0.5,  # Sharpening temperature
    lambdas=[0.1, 0.1, 0.1, 0.1, 0.1],
    device='cpu'
):
    """
    FedCVT training for vertical federated learning
    
    Args:
        A_data: Party A's features (N_A, dim_A), includes aligned + non-aligned
        B_data: Party B's features (N_B, dim_B), includes aligned + non-aligned
        A_labels: Party A's labels (N_A,) or (N_A, 1)
        n_aligned: Number of aligned samples (assumed to be first n_aligned rows)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        repr_dim: Dimension of learned representations
        hidden_dim: Hidden dimension for networks
        t: Probability threshold for pseudo-labels
        T: Temperature for sharpening in attention
        lambdas: Loss weights [λ1, λ2, λ3, λ4, λ5]
        device: 'cpu' or 'cuda'
    
    Returns:
        Trained models (h_A_u, h_A_c, h_B_u, h_B_c, f_A, f_B, f_AB)
    """
    device = torch.device(device)
    
    # Convert to tensors
    if isinstance(A_data, np.ndarray):
        A_data = torch.from_numpy(A_data).float()
    if isinstance(B_data, np.ndarray):
        B_data = torch.from_numpy(B_data).float()
    if isinstance(A_labels, np.ndarray):
        A_labels = torch.from_numpy(A_labels).float()
    if A_labels.dim() == 1:
        A_labels = A_labels.unsqueeze(1)
    
    A_data = A_data.to(device, dtype=torch.float32)
    B_data = B_data.to(device, dtype=torch.float32)
    A_labels = A_labels.to(device, dtype=torch.float32)
    
    dim_A = A_data.shape[1]
    dim_B = B_data.shape[1]
    n_A = A_data.shape[0]
    n_B = B_data.shape[0]
    
    # Split into aligned and non-aligned
    X_A_al = A_data[:n_aligned]
    Y_A_al = A_labels[:n_aligned]
    X_A_nl = A_data[n_aligned:]
    Y_A_nl = A_labels[n_aligned:]
    
    X_B_al = B_data[:n_aligned]
    X_B_nl = B_data[n_aligned:]
    
    # Initialize neural networks
    # Party A: unique and shared representation learners
    h_A_u = RepresentationNet(dim_A, hidden_dim, repr_dim).to(device)
    h_A_c = RepresentationNet(dim_A, hidden_dim, repr_dim).to(device)
    
    # Party B: unique and shared representation learners
    h_B_u = RepresentationNet(dim_B, hidden_dim, repr_dim).to(device)
    h_B_c = RepresentationNet(dim_B, hidden_dim, repr_dim).to(device)
    
    # Attention-based representation estimator
    g = AttentionEstimator(T=T).to(device)
    
    # Classifiers
    f_A = Classifier(repr_dim * 2, hidden_dim=256, num_classes=1).to(device)  # Takes [R_c; R_u]
    f_B = Classifier(repr_dim * 2, hidden_dim=256, num_classes=1).to(device)
    f_AB = Classifier(repr_dim * 4, hidden_dim=256, num_classes=1).to(device)  # Takes [R_A_c; R_A_u; R_B_c; R_B_u]
    
    # Optimizer
    all_params = (list(h_A_u.parameters()) + list(h_A_c.parameters()) +
                  list(h_B_u.parameters()) + list(h_B_c.parameters()) +
                  list(f_A.parameters()) + list(f_B.parameters()) + list(f_AB.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=lr)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    # Training loop
    for epoch in range(epochs):
        # Create batches (simple approach: iterate through aligned samples)
        n_batches = max(1, n_aligned // batch_size)
        
        epoch_losses = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_aligned)
            
            # Sample non-aligned batches
            nl_batch_size = min(batch_size, len(X_A_nl), len(X_B_nl))
            if nl_batch_size > 0:
                nl_indices_A = torch.randperm(len(X_A_nl))[:nl_batch_size]
                nl_indices_B = torch.randperm(len(X_B_nl))[:nl_batch_size]
            else:
                nl_indices_A = torch.tensor([])
                nl_indices_B = torch.tensor([])
            
            # Get batches
            X_A_al_batch = X_A_al[start_idx:end_idx]
            Y_A_al_batch = Y_A_al[start_idx:end_idx]
            X_B_al_batch = X_B_al[start_idx:end_idx]
            
            if len(nl_indices_A) > 0:
                X_A_nl_batch = X_A_nl[nl_indices_A]
                Y_A_nl_batch = Y_A_nl[nl_indices_A]
            else:
                X_A_nl_batch = torch.empty(0, dim_A).to(device)
                Y_A_nl_batch = torch.empty(0, 1).to(device)
            
            if len(nl_indices_B) > 0:
                X_B_nl_batch = X_B_nl[nl_indices_B]
            else:
                X_B_nl_batch = torch.empty(0, dim_B).to(device)
            
            optimizer.zero_grad()
            
            # Step 1: Learn representations
            R_A_u_al = h_A_u(X_A_al_batch)
            R_A_c_al = h_A_c(X_A_al_batch)
            R_A_al = torch.cat([R_A_c_al, R_A_u_al], dim=1)
            
            R_B_u_al = h_B_u(X_B_al_batch)
            R_B_c_al = h_B_c(X_B_al_batch)
            R_B_al = torch.cat([R_B_c_al, R_B_u_al], dim=1)
            
            # All representations for A and B (for attention)
            R_A_c = h_A_c(X_A_al)  # All aligned samples from A
            R_B_c = h_B_c(X_B_al)  # All aligned samples from B
            R_A_u = h_A_u(X_A_al)
            R_B_u = h_B_u(X_B_al)
            
            if len(X_A_nl_batch) > 0:
                R_A_u_nl = h_A_u(X_A_nl_batch)
                R_A_c_nl = h_A_c(X_A_nl_batch)
                R_A_nl = torch.cat([R_A_c_nl, R_A_u_nl], dim=1)
            
            if len(X_B_nl_batch) > 0:
                R_B_u_nl = h_B_u(X_B_nl_batch)
                R_B_c_nl = h_B_c(X_B_nl_batch)
                R_B_nl = torch.cat([R_B_c_nl, R_B_u_nl], dim=1)
            
            # Step 2: Estimate missing representations
            if len(X_B_nl_batch) > 0:
                # Estimate R_A corresponding to X_B_nl (Equations 5 and 6)
                R_A_c_est = g(R_B_c_nl, R_A_c, R_A_c)  # Shared
                R_A_u_est = g(R_B_u_nl, R_B_u, R_A_u)  # Unique
                R_A_est = torch.cat([R_A_c_est, R_A_u_est], dim=1)
            
            if len(X_A_nl_batch) > 0:
                # Estimate R_B corresponding to X_A_nl
                R_B_c_est = g(R_A_c_nl, R_B_c, R_B_c)  # Shared
                R_B_u_est = g(R_A_u_nl, R_A_u, R_B_u)  # Unique
                R_B_est = torch.cat([R_B_c_est, R_B_u_est], dim=1)
            
            # Step 3: Pseudo-label prediction for B's non-aligned samples
            pseudo_labeled_samples = []
            pseudo_labels = []
            
            if len(X_B_nl_batch) > 0:
                with torch.no_grad():
                    # Three classifiers predict
                    logits_A = f_A(R_A_est)
                    logits_B = f_B(R_B_nl)
                    logits_AB = f_AB(torch.cat([R_B_nl, R_A_est], dim=1))
                    
                    # Convert to probabilities
                    probs_A = torch.sigmoid(logits_A)
                    probs_B = torch.sigmoid(logits_B)
                    probs_AB = torch.sigmoid(logits_AB)
                    
                    # Get predictions
                    pred_A = (probs_A > 0.5).float()
                    pred_B = (probs_B > 0.5).float()
                    pred_AB = (probs_AB > 0.5).float()
                    
                    # Filter: keep only if all three agree and probabilities > t
                    agree = (pred_A == pred_B) & (pred_B == pred_AB)
                    high_conf = (probs_A > t) & (probs_B > t) & (probs_AB > t)
                    keep_mask = (agree & high_conf).squeeze()
                    
                    if keep_mask.sum() > 0:
                        pseudo_labeled_samples.append((R_B_nl[keep_mask], R_A_est[keep_mask]))
                        pseudo_labels.append(pred_A[keep_mask])
            
            # Step 4: Cross-view training
            # Training set χ combines aligned + pseudo-labeled samples
            
            # For aligned samples
            R_A_train = R_A_al
            R_B_train = R_B_al
            Y_train = Y_A_al_batch
            
            # Add pseudo-labeled samples if any
            if pseudo_labeled_samples:
                for (R_B_pseudo, R_A_pseudo), Y_pseudo in zip(pseudo_labeled_samples, pseudo_labels):
                    R_A_train = torch.cat([R_A_train, R_A_pseudo], dim=0)
                    R_B_train = torch.cat([R_B_train, R_B_pseudo], dim=0)
                    Y_train = torch.cat([Y_train, Y_pseudo], dim=0)
            
            # Compute classification losses
            L_A_ce = criterion(f_A(R_A_train), Y_train)
            L_B_ce = criterion(f_B(R_B_train), Y_train)
            L_AB_ce = criterion(f_AB(torch.cat([R_B_train, R_A_train], dim=1)), Y_train)
            
            # Step 5: Compute auxiliary losses
            # L_AB_diff: difference between shared representations (Eq 1)
            L_AB_diff = torch.mean((R_A_c_al - R_B_c_al) ** 2)
            
            # L_A_diff: estimated vs actual representations for A (Eq 7)
            # Estimate R_A for aligned samples of B
            R_A_c_al_est = g(R_B_c_al, R_A_c, R_A_c)
            R_A_u_al_est = g(R_B_u_al, R_B_u, R_A_u)
            R_A_al_est = torch.cat([R_A_c_al_est, R_A_u_al_est], dim=1)
            L_A_diff = torch.mean((R_A_al_est - R_A_al) ** 2)
            
            # L_B_diff: estimated vs actual representations for B (Eq 8)
            R_B_c_al_est = g(R_A_c_al, R_B_c, R_B_c)
            R_B_u_al_est = g(R_A_u_al, R_A_u, R_B_u)
            R_B_al_est = torch.cat([R_B_c_al_est, R_B_u_al_est], dim=1)
            L_B_diff = torch.mean((R_B_al_est - R_B_al) ** 2)
            
            # L_sim: orthogonality constraint (Eqs 2, 3)
            L_A_sim = torch.mean((torch.matmul(R_A_u_al, R_A_c_al.T)) ** 2)
            L_B_sim = torch.mean((torch.matmul(R_B_u_al, R_B_c_al.T)) ** 2)
            
            # Total loss (Eq 12)
            L_obj = (L_AB_ce + L_A_ce + L_B_ce + 
                     lambdas[0] * L_AB_diff + 
                     lambdas[1] * L_A_diff + 
                     lambdas[2] * L_B_diff + 
                     lambdas[3] * L_A_sim + 
                     lambdas[4] * L_B_sim)
            
            L_obj.backward()
            optimizer.step()
            
            epoch_losses.append(L_obj.item())
        
        # Print epoch statistics
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {np.mean(epoch_losses):.4f}")
    
    return h_A_u, h_A_c, h_B_u, h_B_c, f_A, f_B, f_AB


# -------------------------
# Evaluation
# -------------------------
def fedcvt_evaluate(h_A_u, h_A_c, h_B_u, h_B_c, f_AB, A_test, B_test, Y_test, device='cpu'):
    """
    Evaluate FedCVT model on test data
    
    Args:
        h_A_u, h_A_c: Party A's representation networks
        h_B_u, h_B_c: Party B's representation networks
        f_AB: Federated classifier
        A_test: Test features from Party A
        B_test: Test features from Party B
        Y_test: Test labels
        device: 'cpu' or 'cuda'
    
    Returns:
        accuracy, predictions
    """
    device = torch.device(device)
    
    # Convert to tensors
    if isinstance(A_test, np.ndarray):
        A_test = torch.from_numpy(A_test).float()
    if isinstance(B_test, np.ndarray):
        B_test = torch.from_numpy(B_test).float()
    if isinstance(Y_test, np.ndarray):
        Y_test = torch.from_numpy(Y_test).float()
    
    A_test = A_test.to(device, dtype=torch.float32)
    B_test = B_test.to(device, dtype=torch.float32)
    Y_test = Y_test.to(dtype=torch.float32).cpu().numpy().ravel()
    
    # Set to eval mode
    h_A_u.eval()
    h_A_c.eval()
    h_B_u.eval()
    h_B_c.eval()
    f_AB.eval()
    
    with torch.no_grad():
        # Get representations
        R_A_u = h_A_u(A_test)
        R_A_c = h_A_c(A_test)
        R_A = torch.cat([R_A_c, R_A_u], dim=1)
        
        R_B_u = h_B_u(B_test)
        R_B_c = h_B_c(B_test)
        R_B = torch.cat([R_B_c, R_B_u], dim=1)
        
        # Federated prediction
        logits = f_AB(torch.cat([R_B, R_A], dim=1))
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs > 0.5).astype(int)
    
    accuracy = accuracy_score(Y_test, preds)
    
    return accuracy, preds
