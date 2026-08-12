import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, d_model):
        super().__init__()
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EEGEncoder(nn.Module):
    def __init__(self, n_channels, seq_len, patch_size, d_model, nhead,
                 num_layers, dim_feedforward, dropout, norm_first):
        super().__init__()
        self.patch_embed = PatchEmbedding(n_channels, seq_len, patch_size, d_model)
        num_patches = seq_len // patch_size
        self.pos_enc = PositionalEncoding(d_model, num_patches + 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=norm_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]


class PretrainModel(nn.Module):
    def __init__(self, encoder, n_channels, seq_len, patch_size, d_model):
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.num_patches = seq_len // patch_size
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, patch_size * n_channels)
        )

    def forward(self, x):
        B = x.shape[0]
        _ = self.encoder(x)
        patches = self.encoder.patch_embed(x)
        N = patches.size(1)
        rec = self.decoder(patches)
        rec = rec.view(B, N, self.n_channels, self.patch_size)
        rec = rec.permute(0, 2, 1, 3).contiguous().view(B, self.n_channels, N * self.patch_size)
        return rec[:, :, :self.seq_len]


class DistillModel(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x_128, x_3):
        with torch.no_grad():
            t_feat = self.teacher(x_128)
        s_feat = self.student(x_3)
        return s_feat, t_feat


class ClassifierModel(nn.Module):
    def __init__(self, encoder, num_classes, d_model):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, num_classes)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.encoder(x)
        out = self.head(feat)
        return out, feat


class DualBranchModel(nn.Module):
    def __init__(self, encoder_general, encoder_class, num_classes, d_model):
        super().__init__()
        self.encoder_general = encoder_general
        self.encoder_class = encoder_class
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(d_model, num_classes)
        )

    def set_general_freeze(self, freeze=True):
        for p in self.encoder_general.parameters():
            p.requires_grad = not freeze

    def set_class_freeze(self, freeze=True):
        for p in self.encoder_class.parameters():
            p.requires_grad = not freeze

    def forward(self, x):
        f_g = self.encoder_general(x)
        f_c = self.encoder_class(x)
        return self.classifier(torch.cat([f_g, f_c], dim=1)), f_g, f_c