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

    def forward(self, x, mask):
        score = self.scorer(x).squeeze(-1)           
        score = score.float()

        score = score.masked_fill(~mask, -1e4)

        w = torch.softmax(score, dim=1).unsqueeze(-1) 
        w = w.to(dtype=x.dtype)                       
        return (x * w).sum(dim=1)                     



class TriStreamModel(nn.Module):
    def __init__(
        self,
        d_ctx_in=512,
        d_face_in=1280,
        d_skel_in=512,
        d=512,
        n_layers=4,
        n_heads=4,
        dropout=0.3,
    ):
        super().__init__()


        self.ctx_in  = nn.Sequential(nn.Linear(d_ctx_in, d), nn.Dropout(dropout))
        self.face_in = nn.Sequential(nn.Linear(d_face_in, d), nn.Dropout(dropout))
        self.skel_in = nn.Sequential(nn.Linear(d_skel_in, d), nn.Dropout(dropout))

        enc = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.ctx_enc  = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.face_enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.skel_enc = nn.TransformerEncoder(enc, num_layers=n_layers)

        self.ctx_pool  = AttnPool(d)
        self.face_pool = AttnPool(d)
        self.skel_pool = AttnPool(d)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )

    def forward(self, ctx, face, skel, time_mask, face_mask, skel_mask):

        pad_ignore = ~time_mask

        x_ctx  = self.ctx_in(ctx)
        x_face = self.face_in(face)
        x_skel = self.skel_in(skel)

        x_ctx  = self.ctx_enc(x_ctx,   src_key_padding_mask=pad_ignore)
        x_face = self.face_enc(x_face, src_key_padding_mask=pad_ignore)
        x_skel = self.skel_enc(x_skel, src_key_padding_mask=pad_ignore)

        ctx_vec  = self.ctx_pool(x_ctx,  time_mask)
        face_vec = self.face_pool(x_face, time_mask & face_mask)
        skel_vec = self.skel_pool(x_skel, time_mask & skel_mask)

        z = torch.cat([ctx_vec, face_vec, skel_vec], dim=-1)
        return self.head(z).squeeze(-1)
