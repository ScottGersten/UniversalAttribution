import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        # x shape: (B, dim) -> unsqueeze to (B, 1, dim) for MultiheadAttention
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
            norm_x = self.norm1(x_seq)
            attn_out, _ = self.attn(norm_x, norm_x, norm_x)
            x_seq = x_seq + attn_out
            x_seq = x_seq + self.mlp(self.norm2(x_seq))
            return x_seq.squeeze(1)
        else:
            norm_x = self.norm1(x)
            attn_out, _ = self.attn(norm_x, norm_x, norm_x)
            x = x + attn_out
            return x + self.mlp(self.norm2(x))

class GenDetAugmenter(nn.Module):
    """
    Warp feature representations during training to harden decision boundaries.
    """
    def __init__(self, dim=768, num_heads=4):
        super().__init__()
        self.block = TransformerBlock(dim=dim, num_heads=num_heads)

    def forward(self, features):
        delta = self.block(features)
        # Residual perturbation
        return features + 0.1 * F.normalize(delta, p=2, dim=-1)

class TeacherStudentGate(nn.Module):
    """
    Computes discrepancy scores:
    Low discrepancy -> Known distribution (passes to Linear Probe)
    High discrepancy -> Flags Unseen Anomaly
    """
    def __init__(self, dim=768, num_heads=4):
        super().__init__()
        self.teacher = TransformerBlock(dim=dim, num_heads=num_heads)
        self.student = TransformerBlock(dim=dim, num_heads=num_heads)
        
        # Freeze teacher by default
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            t_out = self.teacher(x)
        s_out = self.student(x)
        # Discrepancy metric (L2 distance)
        discrepancy = torch.norm(t_out - s_out, p=2, dim=-1)
        return discrepancy, t_out, s_out

    def get_anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            t_out = self.teacher(x)
            s_out = self.student(x)
            return torch.norm(t_out - s_out, p=2, dim=-1)