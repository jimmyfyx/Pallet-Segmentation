import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F
from dpt import DPTHead, resize, HeadDepth, HeadDepthClassi


class DINOv2DPT(nn.Module):
    def __init__(self, output_size=(416, 416), bottleneck_dim=128, output_channels=2, encoder_path=None, decoder_path=None):
        super().__init__()
        self.output_channels = output_channels
        self.output_size = output_size
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.backbone.requires_grad_(False)  # Freeze the encoder
        
        embed_dim = 384
        self.decoder = DPTHead(
            in_channels=[embed_dim] * 4,
            out_channels=output_channels,
            channels=bottleneck_dim,
            embed_dims=embed_dim,
            post_process_channels=[embed_dim // 2 ** (3 - i) for i in range(4)],
            readout_type="project",
        )
        
        self.final_layer = HeadDepth(features=bottleneck_dim, outputs=output_channels)
        self.activation = nn.Sigmoid()  # Regard two masks as not mutually exclusive
        self.align_corners = self.decoder.align_corners

        # # Freeze encoder weights
        # if encoder_path is not None:
        #     self.backbone.load_state_dict(torch.load(encoder_path))
        #     self.backbone.requires_grad_(False)

        # Freeze decoder weights
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path))
            for param in self.decoder.parameters():
                param.requires_grad = False
                pass
            
    def forward(self, x):
        encoding = self.backbone.get_intermediate_layers(
            x,
            n=[2,5,8,11],
            reshape=True,
            return_class_token=True,
            norm=False
        ) # will return (patch_tokens, class_token)
        
        decoding = self.decoder(encoding, img_metas=None)
        decoding = self.activation(self.final_layer(decoding))
        output = resize(input=decoding, size=self.output_size, mode='bilinear', align_corners=self.align_corners)
        if self.output_channels == 1:
            output = output[:,0]
        return output  # (1, output_channels, h, w)
    
# network = DINOv2DPT(output_size=(416, 416), bottleneck_dim=128, output_channels=2)
# predictions = network(torch.randn(1, 3, 420, 420)) # 1 RGB image of shape (364,644)
# print(predictions.size()) # should be (1, 2, 416, 416)