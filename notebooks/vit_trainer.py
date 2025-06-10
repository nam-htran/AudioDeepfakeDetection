# notebooks/vit_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Mặc dù không dùng trực tiếp trong ViT_Audio, nhưng thường có trong notebook
from einops import rearrange 
from einops.layers.torch import Rearrange 

# Các import này có thể không cần thiết cho chỉ định nghĩa lớp ViT_Audio
# nhưng thường có trong notebook huấn luyện đầy đủ. Để an toàn, có thể giữ lại.
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
# import numpy as np
# import pandas as pd
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import warnings
# import wandb # Nếu bạn không dùng wandb khi deploy, không cần import này
# from dataclasses import dataclass


class ViT_Audio(nn.Module):
    def __init__(self, img_size: int, patch_size: int, num_classes: int, 
                 in_channels: int, dim: int, depth: int, heads: int, 
                 mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (img_size // patch_size) ** 2
        patch_dim_flattened = in_channels * patch_size * patch_size 

        self.img_size = img_size # Lưu lại để có thể kiểm tra
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dim = dim

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, patch_dim_flattened, kernel_size=patch_size, stride=patch_size), 
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(patch_dim_flattened),
            nn.Linear(patch_dim_flattened, dim),
            nn.LayerNorm(dim),
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout, 
            batch_first=True,
            activation=F.gelu # Sử dụng F.gelu hoặc 'gelu' tùy phiên bản PyTorch
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)

        # Khôi phục self.ln từ notebook gốc của bạn
        self.ln = nn.LayerNorm(dim) 
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # x có shape (batch_size, in_channels, img_size, img_size)
        # Ví dụ: (B, 1, 224, 224)
        
        x = self.to_patch_embedding(x) # Output: (B, num_patches, dim)
        b, n_patches, _ = x.shape # n_patches ở đây là (img_size/patch_size)**2

        cls_tokens = self.cls_token.expand(b, -1, -1) # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, num_patches + 1, dim)
        
        # Kiểm tra shape của pos_embed một cách cẩn thận
        # num_patches_plus_cls = (self.img_size // self.patch_size)**2 + 1
        # if self.pos_embed.shape[1] != num_patches_plus_cls:
        #     print(f"Warning: pos_embed shape[1] {self.pos_embed.shape[1]} vs expected {num_patches_plus_cls}. Re-initializing pos_embed.")
        #     self.pos_embed = nn.Parameter(torch.randn(1, num_patches_plus_cls, self.dim, device=x.device))

        if self.pos_embed.shape[1] != x.shape[1]:
             # Điều này có thể xảy ra nếu img_size hoặc patch_size thay đổi mà pos_embed không được cập nhật
             # Hoặc lỗi logic trong tính toán num_patches
             # Trong trường hợp deploy, chúng ta thường không muốn tự ý thay đổi pos_embed
             # mà nên đảm bảo các tham số đầu vào là chính xác.
             raise ValueError(f"Shape mismatch for positional embedding: "
                              f"pos_embed has {self.pos_embed.shape[1]} tokens, "
                              f"input has {x.shape[1]} tokens (num_patches+1). "
                              f"Check img_size ({self.img_size}) and patch_size ({self.patch_size}). "
                              f"Calculated num_patches: {n_patches}")
        
        x = x + self.pos_embed # (B, num_patches + 1, dim)
        
        x = self.transformer(x) # Output: (B, num_patches + 1, dim)
        
        cls_token_output = x[:, 0] # Lấy output của CLS token: (B, dim)
        
        # Sử dụng self.ln trước khi đưa vào mlp_head, giống như trong notebook gốc
        processed_cls_token = self.ln(cls_token_output) 
        return self.mlp_head(processed_cls_token)      # Output: (B, num_classes)

# Đoạn code dưới đây chỉ để test, không cần thiết khi deploy và có thể xóa
if __name__ == '__main__':
    print("Testing ViT_Audio class...")

    # Test ViT_Small configuration
    vit_small_params = {
        "img_size": 224, 
        "patch_size": 16,
        "num_classes": 2, 
        "in_channels": 1, 
        "dim": 128,
        "depth": 4, 
        "heads": 4,
        "mlp_dim": 256, 
        "dropout": 0.1
    }
    try:
        model_small = ViT_Audio(**vit_small_params)
        print("\n--- ViT_Audio (Small) ---")
        # print(model_small) # Bỏ comment để xem kiến trúc
        dummy_input_small = torch.randn(2, 
                                  vit_small_params["in_channels"], 
                                  vit_small_params["img_size"], 
                                  vit_small_params["img_size"])
        output_small = model_small(dummy_input_small)
        print(f"Input shape: {dummy_input_small.shape}")
        print(f"Output shape: {output_small.shape}") # Expected (2, num_classes)
        assert output_small.shape == (2, vit_small_params["num_classes"]), "Output shape mismatch for ViT_Small"
        print("ViT_Audio (Small) test passed.")
    except Exception as e:
        print(f"Error testing ViT_Audio (Small): {e}")

    # Test ViT_Large configuration
    vit_large_params = {
        "img_size": 224, 
        "patch_size": 16,
        "num_classes": 2, 
        "in_channels": 1, 
        "dim": 384,
        "depth": 6, 
        "heads": 8,
        "mlp_dim": 768, 
        "dropout": 0.1
    }
    try:
        model_large = ViT_Audio(**vit_large_params)
        print("\n--- ViT_Audio (Large) ---")
        # print(model_large) # Bỏ comment để xem kiến trúc
        dummy_input_large = torch.randn(2, 
                                 vit_large_params["in_channels"], 
                                 vit_large_params["img_size"], 
                                 vit_large_params["img_size"])
        output_large = model_large(dummy_input_large)
        print(f"Input shape: {dummy_input_large.shape}")
        print(f"Output shape: {output_large.shape}") # Expected (2, num_classes)
        assert output_large.shape == (2, vit_large_params["num_classes"]), "Output shape mismatch for ViT_Large"
        print("ViT_Audio (Large) test passed.")
    except Exception as e:
        print(f"Error testing ViT_Audio (Large): {e}")