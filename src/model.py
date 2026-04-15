import torch
import torch.nn as nn

class DeepSetsPredictor(nn.Module):
    def __init__(self, n_props, n_types=10, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, 16, padding_idx=9)
        self.encoder = nn.Sequential(
            nn.Linear(n_props + 1 + 16, emb_dim),  # +16 за тип
            nn.LayerNorm(emb_dim), nn.GELU(), nn.Dropout(0.15)
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim*2 + 4, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU()
        )
        self.vis_head = nn.Linear(hidden_dim//2, 1)
        self.ox_head  = nn.Linear(hidden_dim//2, 1)

    def forward(self, props, mask, conc, conditions, type_ids=None):
        type_ids = torch.zeros(props.shape[0], props.shape[1], dtype=torch.long, device=props.device) if type_ids is None else type_ids
        type_vec = self.type_emb(type_ids)  # [B, N, 16]
        x = torch.cat([props, conc, type_vec], dim=-1)
        h = self.encoder(x) * mask.unsqueeze(-1)
        
        h_global = torch.cat([h.sum(dim=1), h.max(dim=1).values], dim=-1)
        x_in = torch.cat([h_global, conditions], dim=-1)
        shared = self.head(x_in)
        return self.vis_head(shared).squeeze(-1), self.ox_head(shared).squeeze(-1)