import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import default_conv
from .swin_transformer_v2 import BasicLayer, PatchEmbed


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(channels, channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(channels, channels, 3),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualRefinement(nn.Module):
    def __init__(self, channels, depth=2):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.block = nn.Sequential(
            default_conv(in_channels, out_channels * (scale ** 2), 3),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ModalityStem(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(1, out_channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            default_conv(out_channels, out_channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class SobelEdgeExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.project = nn.Sequential(
            default_conv(channels, channels, 3),
            nn.ReLU(inplace=True),
            default_conv(channels, channels, 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, modalities):
        edge_maps = []
        for modality in modalities:
            gx = F.conv2d(modality, self.sobel_x, padding=1)
            gy = F.conv2d(modality, self.sobel_y, padding=1)
            edge_maps.append(torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6))
        return self.project(torch.cat(edge_maps, dim=1))


class TransformerCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, query_dim)
        self.attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, query_map, context_map):
        b, c, h, w = query_map.shape
        query_tokens = query_map.flatten(2).transpose(1, 2)
        context_tokens = context_map.flatten(2).transpose(1, 2)
        query_tokens = self.query_norm(query_tokens)
        context_tokens = self.context_proj(self.context_norm(context_tokens))
        attn_out, _ = self.attn(query_tokens, context_tokens, context_tokens)
        attn_out = self.out_proj(attn_out).transpose(1, 2).reshape(b, c, h, w)
        return query_map + self.scale * attn_out


class FusionContextEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(in_channels, out_channels, 3),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x):
        return self.body(x)


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        target = target.flatten(1)
        intersection = (probs * target).sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (probs.sum(dim=1) + target.sum(dim=1) + self.smooth))
        return bce + dice.mean()


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probs, 1.0 - probs)
        alpha = torch.where(target > 0.5, self.alpha, 1.0 - self.alpha)
        loss = alpha * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


class SwinV2Backbone8x8(nn.Module):
    def __init__(
        self,
        img_size=64,
        in_chans=288,
        patch_size=2,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=4,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        resolution = img_size // patch_size
        drop_path = torch.linspace(0, 0.1, sum(depths)).tolist()

        self.stage1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(resolution, resolution),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[: depths[0]],
            norm_layer=nn.LayerNorm,
            downsample=None,
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.stage2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(resolution // 2, resolution // 2),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[depths[0] : depths[0] + depths[1]],
            norm_layer=nn.LayerNorm,
            downsample=None,
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.stage3 = BasicLayer(
            dim=embed_dim * 4,
            input_resolution=(resolution // 4, resolution // 4),
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[depths[0] + depths[1] : depths[0] + depths[1] + depths[2]],
            norm_layer=nn.LayerNorm,
            downsample=None,
        )
        self.proj4 = nn.Conv2d(embed_dim * 4, embed_dim * 8, kernel_size=1)
        self.stage4 = BasicLayer(
            dim=embed_dim * 8,
            input_resolution=(resolution // 4, resolution // 4),
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[-depths[3] :],
            norm_layer=nn.LayerNorm,
            downsample=None,
        )

    @staticmethod
    def _tokens_to_map(tokens, height, width):
        b, _, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, height, width)

    @staticmethod
    def _map_to_tokens(feature_map):
        return feature_map.flatten(2).transpose(1, 2)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_embed.patch_size[0]
        
        tokens = self.patch_embed(x)
        res1 = h // p
        x_stage1 = self.stage1(tokens)
        feat_stage1 = self._tokens_to_map(x_stage1, res1, res1)

        feat_stage2_in = self.down1(feat_stage1)
        res2 = res1 // 2
        x_stage2 = self.stage2(self._map_to_tokens(feat_stage2_in))
        feat_stage2 = self._tokens_to_map(x_stage2, res2, res2)

        feat_stage3_in = self.down2(feat_stage2)
        res3 = res2 // 2
        x_stage3 = self.stage3(self._map_to_tokens(feat_stage3_in))
        feat_stage3 = self._tokens_to_map(x_stage3, res3, res3)

        feat_stage4_in = self.proj4(feat_stage3)
        res4 = res3
        x_stage4 = self.stage4(self._map_to_tokens(feat_stage4_in))
        feat_stage4 = self._tokens_to_map(x_stage4, res4, res4)

        return {
            "32x32": feat_stage1, # Actually res1 x res1
            "16x16": feat_stage2, # Actually res2 x res2
            "8x8": feat_stage3,   # Actually res3 x res3
            "latent": feat_stage4, # Actually res4 x res4
        }


class SwinMMHCA(nn.Module):
    def __init__(
        self,
        n_inputs=3,
        scale=4,
        height=64,
        width=64,
        patch_size=2,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
    ):
        super().__init__()
        # Relaxing the height/width check for different scales
        if scale == 4:
            if height != 64 or width != 64:
                raise ValueError("For 4x SR, expected 64x64 LR inputs to reach 256x256 HR.")
        elif scale == 2:
            if height != 128 or width != 128:
                raise ValueError("For 2x SR, expected 128x128 LR inputs to reach 256x256 HR.")

        self.n_inputs = n_inputs
        self.scale = scale
        stem_channels = embed_dim

        self.modality_stems = nn.ModuleList([ModalityStem(stem_channels) for _ in range(n_inputs)])
        self.context_encoder = FusionContextEncoder(stem_channels * n_inputs, 256)
        self.edge_branch = SobelEdgeExtractor(n_inputs)
        self.edge_context_fuser = nn.Sequential(
            default_conv(256 + n_inputs, 256, 3),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(256),
        )

        # Backbone input resolution should match current height
        self.backbone = SwinV2Backbone8x8(
            img_size=height,
            in_chans=stem_channels * n_inputs,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=heads,
            window_size=4,
        )

        self.context_to_8 = nn.Sequential(default_conv(256, 384, 3), nn.LeakyReLU(0.2, inplace=True))
        self.context_to_latent = nn.Sequential(default_conv(256, 768, 3), nn.LeakyReLU(0.2, inplace=True))

        self.cross_attn_8 = TransformerCrossAttention(query_dim=384, context_dim=384, num_heads=12)
        self.cross_attn_latent = TransformerCrossAttention(query_dim=768, context_dim=768, num_heads=24)

        self.reduce_latent = nn.Sequential(default_conv(768, 384, 1), nn.LeakyReLU(0.2, inplace=True))
        
        # Decoder structure to reach 256x256
        # Backbone latent is always 1/8 of input resolution.
        # If scale=4: LR=64, Latent=8x8. Need 5 steps (x2 each) to reach 256.
        # If scale=2: LR=128, Latent=16x16. Need 4 steps (x2 each) to reach 256.
        
        self.decoder_8_to_16 = PixelShuffleBlock(384 + 384, 192, scale=2)
        self.decoder_16_to_32 = PixelShuffleBlock(192 + 192, 96, scale=2)
        self.decoder_32_to_64 = PixelShuffleBlock(96 + 96, 64, scale=2)
        self.decoder_64_to_128 = PixelShuffleBlock(64 + 256, 64, scale=2)
        self.decoder_128_to_256 = PixelShuffleBlock(64, 64, scale=2)

        self.reconstruction_head = nn.Sequential(
            ResidualRefinement(64, depth=3),
            default_conv(64, 1, 3),
        )

        self.seg_head_blocks = nn.ModuleList([
            PixelShuffleBlock(384, 192, scale=2),
            PixelShuffleBlock(192, 96, scale=2),
            PixelShuffleBlock(96, 64, scale=2),
            PixelShuffleBlock(64, 32, scale=2),
            PixelShuffleBlock(32, 16, scale=2),
        ])
        # For 4x scale: 5 blocks -> 16 channels. For 2x scale: 4 blocks -> 32 channels.
        self.seg_out_conv = default_conv(16 if scale == 4 else 32, 1, 3)

        self.det_head = nn.Sequential(
            default_conv(768, 256, 3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((5, 5)),
            default_conv(256, 1, 1),
        )

        self.seg_loss = DiceBCELoss()
        self.det_loss = FocalBCELoss()

    def build_targets(self, hr_image):
        seg_target = (hr_image > 0.05).float()

        mean = hr_image.mean(dim=(2, 3), keepdim=True)
        std = hr_image.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        lesion_mask = ((hr_image - mean) > 1.5 * std).float()
        det_target = F.adaptive_max_pool2d(lesion_mask, (5, 5))
        det_target = (det_target > 0.0).float()
        return seg_target, det_target

    def auxiliary_losses(self, seg_logits, det_logits, hr_image):
        seg_target, det_target = self.build_targets(hr_image)
        seg_loss = self.seg_loss(seg_logits, seg_target)
        det_loss = self.det_loss(det_logits, det_target)

        return {
            "seg": seg_loss,
            "det": det_loss,
        }

    def forward(self, x, stage=4):
        if not isinstance(x, (list, tuple)) or len(x) != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} modality tensors.")

        bicubic_base = F.interpolate(x[1], scale_factor=self.scale, mode="bicubic", align_corners=False)

        stem_features = [stem(modality) for stem, modality in zip(self.modality_stems, x)]
        sr_input = torch.cat(stem_features, dim=1)
        context_lr = self.context_encoder(sr_input)
        edge_lr = self.edge_branch(x)
        edge_context = self.edge_context_fuser(torch.cat([context_lr, edge_lr], dim=1))

        backbone_feats = self.backbone(sr_input)
        feat_res1 = backbone_feats["32x32"] # 1/2 input res
        feat_res2 = backbone_feats["16x16"] # 1/4 input res
        feat_res3 = backbone_feats["8x8"]   # 1/8 input res
        latent = backbone_feats["latent"]     # 1/8 input res

        context_res3 = F.interpolate(self.context_to_8(edge_context), size=feat_res3.shape[-2:], mode="bilinear", align_corners=False)
        context_latent = F.interpolate(self.context_to_latent(edge_context), size=latent.shape[-2:], mode="bilinear", align_corners=False)

        if stage >= 2:
            feat_res3 = self.cross_attn_8(feat_res3, context_res3)
            latent = self.cross_attn_latent(latent, context_latent)

        latent_up = self.reduce_latent(latent)
        
        if self.scale == 4:
            # 8 -> 16 -> 32 -> 64 -> 128 -> 256
            # feat_res3 (8x8, 384ch), feat_res2 (16x16, 192ch), feat_res1 (32x32, 96ch)
            x16 = self.decoder_8_to_16(torch.cat([latent_up, feat_res3], dim=1)) # In: 768, Out: 192
            x32 = self.decoder_16_to_32(torch.cat([x16, feat_res2], dim=1))      # In: 384, Out: 96
            x64 = self.decoder_32_to_64(torch.cat([x32, feat_res1], dim=1))      # In: 192, Out: 64
            x128 = self.decoder_64_to_128(torch.cat([x64, context_lr], dim=1))   # In: 320, Out: 64
            x256 = self.decoder_128_to_256(x128)                                 # In: 64,  Out: 64
        else:
            # self.scale == 2
            # 16 -> 32 -> 64 -> 128 -> 256
            # feat_res3 (16x16, 384ch), feat_res2 (32x32, 192ch), feat_res1 (64x64, 96ch)
            # context_lr is 128x128
            x32 = self.decoder_8_to_16(torch.cat([latent_up, feat_res3], dim=1)) # In: 768, Out: 192
            x64 = self.decoder_16_to_32(torch.cat([x32, feat_res2], dim=1))      # In: 384, Out: 96
            x128 = self.decoder_32_to_64(torch.cat([x64, feat_res1], dim=1))     # In: 192, Out: 64
            x256 = self.decoder_64_to_128(torch.cat([x128, context_lr], dim=1))  # In: 320, Out: 64

        residual = self.reconstruction_head(x256)
        sr = torch.clamp(bicubic_base + residual, 0.0, 1.0)

        # Dynamic Segmentation Head
        seg_feat = feat_res3 if stage >= 2 else backbone_feats["8x8"]
        # If scale=4: seg_feat is 8x8. Need all 5 blocks (0,1,2,3,4) to reach 256.
        # If scale=2: seg_feat is 16x16. Need only 4 blocks (1,2,3,4) to reach 256.
        # Note: blocks[0] takes 384ch, blocks[1] takes 192ch.
        # So for scale=2, we need to skip the first upsample block but keep the channel reduction if needed,
        # OR better, just use the blocks that match the current resolution.
        
        if self.scale == 4:
            for block in self.seg_head_blocks:
                seg_feat = block(seg_feat)
        else:
            # scale == 2, seg_feat is 16x16 with 384 channels.
            # We must use blocks[1:] resolutions but the first block in the sequence 
            # MUST handle the 384 input channels.
            # Let's simplify: slice the blocks but ensure the first one matches feat_res3 channels.
            # Actually, feat_res3 is ALWAYS 384 channels in this backbone.
            # So for scale 2, we just need fewer upsamples. 
            # We'll use blocks[0] (384->192, 16->32), then blocks[2:] to skip one x2.
            # Wait, 16->32->64->128->256 is 4 steps. 
            # Steps: 
            # 1. 16->32 (block 0: 384->192)
            # 2. 32->64 (block 1: 192->96)
            # 3. 64->128 (block 2: 96->64)
            # 4. 128->256 (block 3: 64->32) -> Wait, block 3 is 64->32, block 4 is 32->16.
            
            # Let's just iterate and stop when resolution hits 256.
            curr_res = seg_feat.shape[-1]
            for block in self.seg_head_blocks:
                if curr_res >= 256:
                    break
                seg_feat = block(seg_feat)
                curr_res = seg_feat.shape[-1]
        
        seg_logits = self.seg_out_conv(seg_feat)
        det_logits = self.det_head(latent)

        if stage < 3:
            seg_logits = torch.zeros_like(seg_logits)
            det_logits = torch.zeros_like(det_logits)

        return {
            "sr": sr,
            "seg_logits": seg_logits,
            "det_logits": det_logits,
            "edge_features": edge_lr,
            "context_features": edge_context,
        }
