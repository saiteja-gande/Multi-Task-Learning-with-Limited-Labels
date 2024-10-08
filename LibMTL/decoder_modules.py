import torch
import torch.nn as nn
import torch.nn.functional as F


class Transform(nn.Module):
    """
    Transform function
    :param tuple input_size: Input feature size
    :param list in_dims: List of nput feature dimensions
    :param int embed_dim: Embedding dimension
    """

    def __init__(self, input_size, in_dims, embed_dim):
        super().__init__()
        assert len(in_dims) == 4  # features from four encoder layers
        # [(H,W), (H/2,W/2), (H/4,W/4), (H/8,W/8)]
        self.feature_sizes = [(input_size[0] * 2**(3 - i), input_size[1] * 2**(3 - i)) for i in range(4)]
        self.in_dims = in_dims
        self.embed_dim = embed_dim

        self.linears = nn.ModuleList()
        for dim in in_dims:
            self.linears.append(nn.Linear(dim, embed_dim))

        self.linear_fuse = nn.Sequential(nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim),
                                         nn.LeakyReLU(inplace=True))

    def forward(self, inputs):
        B = inputs[0].shape[0]

        feas = []
        for i in range(4):
            # print(f'the shape {len(inputs.shape)} at {i}')
            if len(inputs[i].shape) == 4:
                # B, C, h, w
                assert inputs[i].shape[1] == self.in_dims[i], "Input feature dimension mismatch!"
                fea = inputs[i].reshape(B, self.in_dims[i], -1).permute(0, 2, 1)  # B, h*w, C
            elif len(inputs[i].shape) == 3:
                # B, h*w, C
                assert inputs[i].shape[2] == self.in_dims[i], "Input feature dimension mismatch!"
                fea = inputs[i]
            else:
                raise ValueError

            # dimension reduction
            fea = self.linears[i](fea)
            # B, h*w, C => B, C, h*w => B, C, h, w
            fea = fea.permute(0, 2, 1).reshape(B, self.embed_dim, self.feature_sizes[i][0], self.feature_sizes[i][1])
            # B, C, h, w => B, C, H/4, W/4
            fea = F.interpolate(fea, size=self.feature_sizes[0], mode='bilinear', align_corners=False)
            feas.append(fea)

        # B, 4*C, H/4, W/4 => B, C, H/4, W/4
        x = self.linear_fuse(torch.cat(feas, dim=1).contiguous())

        return x
