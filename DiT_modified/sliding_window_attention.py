import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention

class SlidingWindowAttention(Attention):
    """
    Implementation of Sliding Window Attention that restricts attention to a local window.
    Based on timm's Attention module but modified for local window attention.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=8):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.window_size = window_size
        
    def forward(self, x):
        if self.window_size % 2 == 0:
            raise ValueError(f"Sliding window attention requires an odd window_size for symmetric padding; got {self.window_size}")
        B, N, C = x.shape
        # Assuming input is from a square feature map
        H = W = int(N ** 0.5)
        
        # Original qkv projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        
        # Reshape to include spatial dimensions
        q = q.reshape(B, self.num_heads, H, W, C // self.num_heads)
        k = k.reshape(B, self.num_heads, H, W, C // self.num_heads)
        v = v.reshape(B, self.num_heads, H, W, C // self.num_heads)
        
        # Sliding window attention
        # We'll implement this using unfold to create local windows
        padding = self.window_size // 2
        
        # Pad feature maps for sliding window
        q_pad = F.pad(q.permute(0, 1, 4, 2, 3), (padding, padding, padding, padding)).permute(0, 1, 3, 4, 2)
        k_pad = F.pad(k.permute(0, 1, 4, 2, 3), (padding, padding, padding, padding)).permute(0, 1, 3, 4, 2)
        v_pad = F.pad(v.permute(0, 1, 4, 2, 3), (padding, padding, padding, padding)).permute(0, 1, 3, 4, 2)
        
        # Extract windows using unfold
        windows_q = []
        windows_k = []
        windows_v = []
        
        for i in range(H):
            for j in range(W):
                # Extract window for each position
                window_q = q_pad[:, :, i:i+self.window_size, j:j+self.window_size, :]  # B, heads, window_size, window_size, C
                window_k = k_pad[:, :, i:i+self.window_size, j:j+self.window_size, :]
                window_v = v_pad[:, :, i:i+self.window_size, j:j+self.window_size, :]
                
                # Reshape to attention format
                window_q = window_q.reshape(B, self.num_heads, self.window_size*self.window_size, -1)  # B, heads, window_size^2, head_dim
                window_k = window_k.reshape(B, self.num_heads, self.window_size*self.window_size, -1)
                window_v = window_v.reshape(B, self.num_heads, self.window_size*self.window_size, -1)
                
                windows_q.append(window_q)
                windows_k.append(window_k)
                windows_v.append(window_v)
        
        # Process each window with attention
        outputs = []
        for i, (window_q, window_k, window_v) in enumerate(zip(windows_q, windows_k, windows_v)):
            # Calculate attention scores
            attn = (window_q @ window_k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # Apply attention to values
            output = (attn @ window_v).reshape(B, self.num_heads, 1, -1)  # B, heads, 1, head_dim
            outputs.append(output)
        
        # Concatenate outputs and reshape
        x = torch.cat(outputs, dim=2)  # B, heads, N, head_dim
        x = x.transpose(1, 2).reshape(B, N, C)  # B, N, C
        
        # Project back to output dimension
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

# More efficient implementation using unfold operation
class EfficientSlidingWindowAttention(Attention):
    """
    More efficient implementation of Sliding Window Attention using unfold.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=8):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.window_size = window_size
        
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        
        # Original QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        
        # Reshape for spatial processing
        q = q.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads)
        k = k.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads)
        v = v.transpose(1, 2).reshape(B, N, self.num_heads, C // self.num_heads)
        
        # Reshape to 2D spatial layout
        q = q.reshape(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # B, heads, H, W, C_head
        k = k.reshape(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(B, H, W, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        
        # Create sliding window attention using unfold
        padding = self.window_size // 2
        
        # Use unfold for efficient sliding window
        q_unfolded = F.unfold(q.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2), 
                             kernel_size=self.window_size, padding=padding)
        k_unfolded = F.unfold(k.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2),
                             kernel_size=self.window_size, padding=padding)
        v_unfolded = F.unfold(v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2),
                             kernel_size=self.window_size, padding=padding)
        
        # Reshape unfolded tensors
        q_unfolded = q_unfolded.reshape(B, self.num_heads, -1, self.window_size*self.window_size, N)
        k_unfolded = k_unfolded.reshape(B, self.num_heads, -1, self.window_size*self.window_size, N)
        v_unfolded = v_unfolded.reshape(B, self.num_heads, -1, self.window_size*self.window_size, N)
        
        # Compute attention scores
        attn = torch.einsum('bhcwn,bhcwm->bhnwm', q_unfolded, k_unfolded) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = torch.einsum('bhnwm,bhcwm->bhcn', attn, v_unfolded)
        
        # Reshape to original format
        out = out.reshape(B, self.num_heads, C // self.num_heads, H*W).permute(0, 3, 1, 2).reshape(B, N, C)
        
        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out