import torch
import torch.nn as nn
import torch.nn.functional as F

# class WBCPriorLayer(nn.Module):
#     def __init__(self, kernel_size, typeprior='DARK'):
#         super(WBCPriorLayer, self).__init__()
#         self.kernel_size = kernel_size
#         self.pad_h = self.pad_w = kernel_size // 2
#         self.typeprior = typeprior
#
#     def forward(self, x):
#         n, c, h, w = x.size()
#         pooled_height = h
#         pooled_width = w
#         top_data = torch.zeros((n, 1, pooled_height, pooled_width), dtype=x.dtype, device=x.device)
#         mask = torch.zeros((n, 1, pooled_height, pooled_width), dtype=torch.int64, device=x.device)
#
#         if self.typeprior == 'DARK':
#             top_data.fill_(float('inf'))
#         elif self.typeprior == 'WHITE':
#             top_data.fill_(float('-inf'))
#         else:
#             raise ValueError("Unknown prior method")
#
#         for n_idx in range(n):
#             for ph in range(pooled_height):
#                 for pw in range(pooled_width):
#                     hstart = max(ph - self.pad_h, 0)
#                     wstart = max(pw - self.pad_w, 0)
#                     hend = min(ph - self.pad_h + self.kernel_size, h)
#                     wend = min(pw - self.pad_w + self.kernel_size, w)
#
#                     for c_idx in range(c):
#                         for h_idx in range(hstart, hend):
#                             for w_idx in range(wstart, wend):
#                                 index = c_idx * h * w + h_idx * w + w_idx
#                                 if self.typeprior == 'DARK' and x[n_idx, c_idx, h_idx, w_idx] < top_data[n_idx, 0, ph, pw]:
#                                     top_data[n_idx, 0, ph, pw] = x[n_idx, c_idx, h_idx, w_idx]
#                                     mask[n_idx, 0, ph, pw] = index
#                                 elif self.typeprior == 'WHITE' and x[n_idx, c_idx, h_idx, w_idx] > top_data[n_idx, 0, ph, pw]:
#                                     top_data[n_idx, 0, ph, pw] = x[n_idx, c_idx, h_idx, w_idx]
#                                     mask[n_idx, 0, ph, pw] = index
#
#         return top_data, mask


class WBCPriorLayer(nn.Module):
    def __init__(self, kernel_size):
        super(WBCPriorLayer, self).__init__()
        self.kernel_size = kernel_size
        self.pad_h = self.pad_w = kernel_size // 2
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=self.pad_h)

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded = self.unfold(x)  # Shape: (n, c*kernel_size*kernel_size, L)
        top_data, idx = unfolded.min(dim=1, keepdim=True)
        top_data = top_data.view(n, 1, h, w)
        mask = idx.view(n, 1, h, w)
        # unfolded = unfolded.view(n, c, self.kernel_size, self.kernel_size, -1)
        # unfolded = unfolded.permute(0, 4, 1, 2, 3)  # Shape: (n, L, c, kernel_size, kernel_size)
        #
        # if self.typeprior == 'DARK':
        #     top_data, idx = unfolded.min(dim=(2, 3, 4))
        # elif self.typeprior == 'WHITE':
        #     top_data, idx = unfolded.max(dim=(2, 3, 4))
        # else:
        #     raise ValueError("Unknown prior method")
        #
        # top_data = top_data.view(n, 1, h, w)
        # mask = idx.view(n, 1, h, w)

        return top_data, mask



if __name__ == '__main__':
    layer = WBCPriorLayer(kernel_size=5)
    x = torch.randn(1, 3, 10, 10, requires_grad=True, device='cuda')  # Ensure the input tensor is on GPU
    print(x)
    output, mask = layer(x)
    print(output)
    norm1 = torch.norm(output, p=1)
    print(norm1)
    print(norm1.requires_grad)