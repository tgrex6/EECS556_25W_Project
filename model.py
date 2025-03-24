import torch
import torch.nn as nn
from functools import partial
from wavevit import Block, WaveViT

class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks, num_heads, mlp_ratio, sr_ratio):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fusion_conv = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            Block(
                dim=out_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=0., 
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                sr_ratio=sr_ratio,
                block_type='wave'
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion_conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        for block in self.blocks:
            x = block(x, H, W)

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks, num_heads, mlp_ratio, sr_ratio):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.has_skip = skip_channels > 0
        if self.has_skip:
            self.fusion_conv = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        else:
            self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            Block(
                dim=out_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                sr_ratio=sr_ratio,
                block_type='wave'
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, skip):
        x = self.upsample(x)
        
        if self.has_skip and skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        x = self.fusion_conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        for block in self.blocks:
            x = block(x, H, W)
        
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

class DenoiseDecoder(nn.Module):
    def __init__(self, 
                 encoder_channels=[64, 128, 320, 448],
                 decoder_channels=[448, 320, 128, 64],
                 num_blocks=[2, 2, 2, 1],
                 num_heads=[14, 10, 4, 2],
                 mlp_ratios=[4, 4, 8, 8],
                 sr_ratios=[1, 1, 2, 2]):
        super().__init__()
        
        self.stages = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[-1] if i == 0 else decoder_channels[i-1]
            skip_ch = encoder_channels[-2-i] if (i < len(encoder_channels)-1) else 0
            
            self.stages.append(
                DecoderStage(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=decoder_channels[i],
                    num_blocks=num_blocks[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    sr_ratio=sr_ratios[i]
                )
            )
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        x = features[-1]
        skips = features[:-1][::-1]
        
        for i, stage in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            x = stage(x, skip)
        
        return self.final_conv(x)




if __name__ == "__main__":
    encoder = WaveViT()
    decoder = DenoiseDecoder()

    noisy_img = torch.randn(2, 3, 224, 224)  # Example input tensor

    features = encoder(noisy_img)

    denoised_img = decoder(features)
    print(denoised_img.shape)  # Should output torch.Size([2, 3, 224, 224])
