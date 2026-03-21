import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .swin_transformer_v2 import SwinTransformer
from .mhca import MHCA
from .common import default_conv

# --- 1. Edge Extraction Module ---
class EdgeModule(nn.Module):
    def __init__(self, in_channels, n_feats):
        super(EdgeModule, self).__init__()
        self.register_buffer('filter_h', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3))
        self.register_buffer('filter_v', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3))
        
        self.init_conv = default_conv(in_channels, n_feats, 3)
        self.texture_branch = nn.Sequential(
            default_conv(n_feats, n_feats, 3),
            nn.LeakyReLU(0.2, True),
            default_conv(n_feats, n_feats, 3),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self, x):
        if isinstance(x, list): x = torch.cat(x, dim=1)
        B, C, H, W = x.shape
        fh = self.filter_h.repeat(C, 1, 1, 1)
        fv = self.filter_v.repeat(C, 1, 1, 1)
        grad_h = F.conv2d(x, fh, padding=1, groups=C)
        grad_v = F.conv2d(x, fv, padding=1, groups=C)
        edge_map = torch.sqrt(grad_h**2 + grad_v**2 + 1e-6)
        return self.texture_branch(self.init_conv(edge_map))

# --- 2. Improved Cross-Attention Fusion ---
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, context_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query = default_conv(dim, dim, 1)
        self.key = default_conv(context_dim, dim, 1)
        self.value = default_conv(context_dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = default_conv(dim, dim, 3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, sr_feat, context_feat):
        B, C, H, W = sr_feat.shape
        ctx = F.interpolate(context_feat, size=(H, W), mode='bilinear', align_corners=False)
        q = self.query(sr_feat).view(B, C, -1)
        k = self.key(ctx).view(B, C, -1)
        v = self.value(ctx).view(B, C, -1)
        attn = self.softmax(torch.bmm(q.permute(0, 2, 1), k))
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return sr_feat + self.gamma * self.proj(out)

class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(default_conv(n_feats, n_feats, 3), nn.LeakyReLU(0.2, True), default_conv(n_feats, n_feats, 3))
    def forward(self, x): return x + self.body(x)

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Sequential(
            default_conv(in_channels, out_channels * (scale_factor ** 2), 3),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(0.2, True)
        )
        self.refine = ResidualBlock(out_channels)
    def forward(self, x): return self.refine(self.upsample(x))

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        pe = torch.zeros(d_model, height, width)
        d_model_half = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        pos_w = torch.arange(0., width).unsqueeze(1); pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model_half+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe

class SuperResolutionPath(nn.Module):
    def __init__(self, n_inputs, n_feats, height, width):
        super(SuperResolutionPath, self).__init__()
        self.cnn_encoders = nn.ModuleList([nn.Sequential(default_conv(1, n_feats, 3), nn.LeakyReLU(0.2, True)) for _ in range(n_inputs)])
        self.pos_encoder = PositionalEncoding2D(d_model=n_feats * n_inputs, height=height, width=width)
        self.transformer = SwinTransformer(
            img_size=height, patch_size=2, in_chans=n_feats * n_inputs,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=4
        )
    def forward(self, x):
        x_cnn = torch.cat([encoder(xi) for xi, encoder in zip(x, self.cnn_encoders)], dim=1) if isinstance(x, list) else self.cnn_encoders[0](x)
        features = self.transformer.forward_features(self.pos_encoder(x_cnn))
        ms_features = [x_cnn] # 0:64x64
        for f in features:
            B, L, C = f.shape
            H_swin = W_swin = int(L**0.5)
            ms_features.append(f.view(B, H_swin, W_swin, C).permute(0, 3, 1, 2))
        return ms_features # [64, 32, 16, 8, 4, 4]

class MedicalContextPath(nn.Module):
    def __init__(self, n_inputs, n_feats):
        super(MedicalContextPath, self).__init__()
        self.encoder = nn.Sequential(default_conv(n_inputs, n_feats, 3), nn.LeakyReLU(0.2, True))
        self.mhca = MHCA(n_feats=n_feats, ratio=4)
        self.edge_refiner = default_conv(n_feats * 2, n_feats, 3)
        self.seg_decoder = nn.Sequential(UpsampleConvLayer(n_feats, n_feats, 2), UpsampleConvLayer(n_feats, 32, 2), default_conv(32, 1, 3))
    def forward(self, x, edge_feats):
        x_in = torch.cat(x, dim=1) if isinstance(x, list) else x
        x_enc = self.encoder(x_in)
        x_ctx = self.mhca(self.edge_refiner(torch.cat([x_enc, edge_feats], dim=1)))
        return x_ctx, self.seg_decoder(x_ctx)

class SwinMMHCA(nn.Module):
    def __init__(self, n_inputs=3, n_feats=64, scale=4, height=64, width=64):
        super(SwinMMHCA, self).__init__()
        self.edge_module = EdgeModule(n_inputs, n_feats)
        self.path_sr = SuperResolutionPath(n_inputs, n_feats, height, width)
        self.path_context = MedicalContextPath(n_inputs, n_feats)
        
        # Decoder stages matching patch_size=2
        self.fusion_8x8 = CrossAttentionFusion(dim=384, context_dim=n_feats)
        self.fusion_16x16 = CrossAttentionFusion(dim=192, context_dim=n_feats)
        
        self.up_4to8 = UpsampleConvLayer(768, 384, 2)
        self.up_8to16 = UpsampleConvLayer(384 + 384, 192, 2)
        self.up_16to32 = UpsampleConvLayer(192 + 192, 96, 2)
        self.up_32to64 = UpsampleConvLayer(96 + 96, 64, 2)
        self.up_64to128 = UpsampleConvLayer(64 + 64, 64, 2)
        self.up_128to256 = UpsampleConvLayer(64, 64, 2)
        
        self.refine_256 = nn.Sequential(ResidualBlock(64), ResidualBlock(64), default_conv(64, 1, 3))
        self.det_head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 64 * 64, 512), nn.LeakyReLU(0.2, True), nn.Linear(512, 5 * 5))

    def forward(self, x):
        input_t2 = x[1] if isinstance(x, list) else x
        bicubic_base = F.interpolate(input_t2, scale_factor=4, mode='bicubic', align_corners=False)
        
        edge_feats = self.edge_module(x)
        ms_feats = self.path_sr(x) # [64, 32, 16, 8, 4, 4]
        context_feats, seg_mask = self.path_context(x, edge_feats)
        
        # Guided Decoding with explicit scale verification
        x_up = self.up_4to8(ms_feats[4]) # 4->8
        x_up = torch.cat([x_up, self.fusion_8x8(ms_feats[3], context_feats)], dim=1) # 8+8
        
        x_up = self.up_8to16(x_up) # 8->16
        x_up = torch.cat([x_up, self.fusion_16x16(ms_feats[2], context_feats)], dim=1) # 16+16
        
        x_up = self.up_16to32(x_up) # 16->32
        x_up = torch.cat([x_up, ms_feats[1]], dim=1) # 32+32
        
        x_up = self.up_32to64(x_up) # 32->64
        x_up = torch.cat([x_up, context_feats], dim=1) # 64+64
        
        x_up = self.up_128to256(self.up_64to128(x_up))
        hr_image = torch.clamp(bicubic_base + self.refine_256(x_up), 0, 1)
        
        return hr_image, seg_mask, self.det_head(context_feats).view(-1, 5, 5)
