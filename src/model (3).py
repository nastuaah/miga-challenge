import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1)
        )

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d)
        mask: (batch, seq_len) булева маска, True для валидных токенов
        """
        if mask is None:
            return x.mean(dim=1)

        if mask.numel() == 0 or mask.sum() == 0:
            return torch.zeros(x.size(0), x.size(2), device=x.device, dtype=x.dtype)

        score = self.scorer(x).squeeze(-1)
        score = score.float()
        score = score.masked_fill(~mask, -1e4)
        w = torch.softmax(score, dim=1).unsqueeze(-1)
        w = w.to(dtype=x.dtype)
        return (x * w).sum(dim=1)

class ContextModel(nn.Module):
    def __init__(self, d_ctx_in, d=256, n_layers=4, n_heads=4, dropout=0.2):
        super().__init__()
        self.ctx_in = nn.Sequential(nn.Linear(d_ctx_in, d), nn.Dropout(dropout))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.ctx_enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.ctx_pool = AttnPool(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, ctx):
        x_ctx = self.ctx_in(ctx)
        x_ctx = self.ctx_enc(x_ctx)
        ctx_vec = self.ctx_pool(x_ctx)
        return self.head(ctx_vec).squeeze(-1)

class FaceModel(nn.Module):
    def __init__(self, d_face_in, d=256, n_layers=4, n_heads=4, dropout=0.2):
        super().__init__()
        self.face_in = nn.Sequential(nn.Linear(d_face_in, d), nn.Dropout(dropout))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.face_enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.face_pool = AttnPool(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, face):
        x_face = self.face_in(face)
        x_face = self.face_enc(x_face)
        face_vec = self.face_pool(x_face)
        return self.head(face_vec).squeeze(-1)

class SkelModel(nn.Module):
    def __init__(self, d_skel_in, d=256, n_layers=4, n_heads=4, dropout=0.2):
        super().__init__()
        self.skel_in = nn.Sequential(nn.Linear(d_skel_in, d), nn.Dropout(dropout))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=4*d,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.skel_enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.skel_pool = AttnPool(d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, skel):
        x_skel = self.skel_in(skel)
        x_skel = self.skel_enc(x_skel)
        skel_vec = self.skel_pool(x_skel)
        return self.head(skel_vec).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets.long()]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def create_optimizer(model, lr=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    return optimizer
