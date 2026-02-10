import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .swin_transformer_v2 import SwinTransformer
from .mhca import MHCA
from .common import default_conv

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd d_model_dim.")
        self.d_model = d_model
        pe = torch.zeros(d_model, height, width)
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Sequential(
            default_conv(in_channels, out_channels * (scale_factor ** 2), 3),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(True),
            default_conv(out_channels, out_channels, 3),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.upsample(x)

class SwinMMHCA(nn.Module):
    def __init__(self, n_inputs=1, n_feats=64, scale=4, height=64, width=64):
        super(SwinMMHCA, self).__init__()
        
        self.n_inputs = n_inputs
        
        self.cnn_encoders = nn.ModuleList([
            nn.Sequential(
                default_conv(1, n_feats, 3),
                nn.ReLU(True)
            ) for _ in range(n_inputs)
        ])
        
        self.pos_encoder = PositionalEncoding2D(d_model=n_feats * n_inputs, height=height, width=width)
        
        self.transformer_encoder = SwinTransformer(
            img_size=height,
            patch_size=4,
            in_chans=n_feats * n_inputs,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4
        )
        
        self.mmhca = MHCA(n_feats=384, ratio=4)

        self.skip_conv = default_conv(n_feats * n_inputs, 192, 1)

        # Decoder for SwinMMHCA with PixelShuffle and Skip Connections
        # Assuming H_swin=W_swin=4 (output from MHCA)
        # Upsample 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 (4 UpsampleConvLayer)
        # Here we concatenate with skip connection from cnn_encoders (64x64)
        # Upsample 64x64 -> 128x128 -> 256x256 (2 more UpsampleConvLayer)

        self.upsample_block1 = UpsampleConvLayer(384, 192, scale_factor=2) # 4x4 -> 8x8
        self.upsample_block2 = UpsampleConvLayer(192, 192, scale_factor=2) # 8x8 -> 16x16
        self.upsample_block3 = UpsampleConvLayer(192, 192, scale_factor=2) # 16x16 -> 32x32
        self.upsample_block4 = UpsampleConvLayer(192, 192, scale_factor=2) # 32x32 -> 64x64

        # After upsample_block4, feature map is (B, 192, 64, 64)
        # Skip connection from cnn_encoders has channels n_feats * n_inputs.
        # We process this and concatenate here.

        self.conv_concat_skip = default_conv(192 + 192, 192, 3) # 192 (from decoder) + 192 (from skip) -> 192
        
        self.upsample_block5 = UpsampleConvLayer(192, 192, scale_factor=2) # 64x64 -> 128x128
        self.upsample_block6 = UpsampleConvLayer(192, 192, scale_factor=2) # 128x128 -> 256x256

        self.final_conv = default_conv(192, 1, 3)


    def forward(self, x):
        if isinstance(x, list):
            # Multi-input case
            encoded_features_list = [encoder(xi) for xi, encoder in zip(x, self.cnn_encoders)]
            # Capture skip connection before positional encoding and transformer
            x_encoder_skip = torch.cat(encoded_features_list, dim=1) # (B, n_feats*n_inputs, H, W)
            x = x_encoder_skip
        else:
            # Single-input case
            x_encoder_skip = self.cnn_encoders[0](x) # (B, n_feats*n_inputs, H, W)
            x = x_encoder_skip
            
        x = self.pos_encoder(x)
        
        features = self.transformer_encoder.forward_features(x)
        
        x_p0 = features[1] 
        
        B, L, C = x_p0.shape
        H_swin = W_swin = int(L**0.5) # This should resolve to 4x4
        x = x_p0.view(B, H_swin, W_swin, C).permute(0, 3, 1, 2) # (B, 384, H_swin, W_swin)
        
        x = self.mmhca(x) # Input to first UpsampleConvLayer is (B, 384, 4, 4)

        # Decoder path with PixelShuffle and Skip Connection
        x = self.upsample_block1(x) # (B, 192, 8, 8)
        x = self.upsample_block2(x) # (B, 192, 16, 16)
        x = self.upsample_block3(x) # (B, 192, 32, 32)
        x = self.upsample_block4(x) # (B, 192, 64, 64) - This is where the skip connection should be added

        # Prepare and apply skip connection
        skip_features = self.skip_conv(x_encoder_skip) # (B, 192, 64, 64)
        x = torch.cat((x, skip_features), dim=1) # Concatenate features along channel dimension
        x = self.conv_concat_skip(x) # (B, 192, 64, 64)

        x = self.upsample_block5(x) # (B, 192, 128, 128)
        x = self.upsample_block6(x) # (B, 192, 256, 256)
        
        x = self.final_conv(x)
        return x

if __name__ == '__main__':
    # Example of how to use the model
    # Single-input case
    print("Testing single-input case...")
    model_single = SwinMMHCA(n_inputs=1)
    input_single = torch.randn(1, 1, 64, 64)
    output_single = model_single(input_single)
    print(f"Output shape (single-input): {output_single.shape}")
    
    # Multi-input case
    print("\nTesting multi-input case...")
    model_multi = SwinMMHCA(n_inputs=3)
    input1 = torch.randn(1, 1, 64, 64)
    input2 = torch.randn(1, 1, 64, 64)
    input3 = torch.randn(1, 1, 64, 64)
    output_multi = model_multi([input1, input2, input3])
    print(f"Output shape (multi-input): {output_multi.shape}")