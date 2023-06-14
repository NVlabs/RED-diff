import torch
import torch.nn as nn

from .dct import LinearDCT, apply_linear_2d
from .jpeg_quantization import quantization_matrix


def torch_rgb2ycbcr(x):
    # Assume x is a batch of size (N x C x H x W)
    v = torch.tensor([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]).to(x.device)
    ycbcr = torch.tensordot(x, v, dims=([1], [1])).transpose(3, 2).transpose(2, 1)
    ycbcr[:,1:] += 128
    return ycbcr


def torch_ycbcr2rgb(x):
    # Assume x is a batch of size (N x C x H x W)
    v = torch.tensor([[ 1.00000000e+00, -3.68199903e-05,  1.40198758e+00],
       [ 1.00000000e+00, -3.44113281e-01, -7.14103821e-01],
       [ 1.00000000e+00,  1.77197812e+00, -1.34583413e-04]]).to(x.device)
    x[:, 1:] -= 128
    rgb = torch.tensordot(x, v, dims=([1], [1])).transpose(3, 2).transpose(2, 1)
    return rgb

def chroma_subsample(x):
    return x[:, 0:1, :, :], x[:, 1:, ::2, ::2]


def jpeg_project(x, y, qf=10):
    # TODO: any size, quantization
    # Assume x is a batch of size (N x C x H x W)
    # [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255
    n_batch, _, n_size, _ = x.shape
    
    x = torch_rgb2ycbcr(x)
    x_chroma_orig = x[:, 1:, :, :].clone()
    x_luma, x_chroma = chroma_subsample(x)
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_chroma = unfold(x_chroma).transpose(2, 1)

    x_luma = x_luma.reshape(-1, 8, 8) - 128
    x_chroma = x_chroma.reshape(-1, 8, 8) - 128
    
    dct_layer = LinearDCT(8, 'dct', norm='ortho')
    dct_layer.to(x.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)
    
    x_luma = x_luma.view(-1, 1, 8, 8)
    x_chroma = x_chroma.view(-1, 2, 8, 8)

    #return y to the DCT shape
    y_luma, y_chroma = y
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    y_luma = unfold(y_luma).transpose(2, 1)
    y_luma = y_luma.reshape(-1, 1, 8, 8)
    y_chroma = unfold(y_chroma).transpose(2, 1)
    y_chroma = y_chroma.reshape(-1, 2, 8, 8)

    #project x to the quantization bounds of y
    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x.device)
    q2 = q2.to(x.device)
    x_luma /= q1.view(1, 8, 8)
    x_chroma /= q2.view(1, 8, 8)

    #print(((x < y - 0.5) | (x > y + 0.5)).sum() / (x <= 1000).sum())
    #print(((x < y - 0.5) | (x > y + 0.5)).sum(dim=0)[0, :, :])
    
    x_luma[x_luma < y_luma - 0.5] = y_luma[x_luma < y_luma - 0.5] - 0.5
    x_luma[x_luma > y_luma + 0.5] = y_luma[x_luma > y_luma + 0.5] + 0.5
    
    x_chroma[x_chroma < y_chroma - 0.5] = y_chroma[x_chroma < y_chroma - 0.5] - 0.5
    x_chroma[x_chroma > y_chroma + 0.5] = y_chroma[x_chroma > y_chroma + 0.5] + 0.5

    x_luma *= q1.view(1, 8, 8)
    x_chroma *= q2.view(1, 8, 8)

    #decode x
    x_luma = x_luma.reshape(-1, 8, 8)
    x_chroma = x_chroma.reshape(-1, 8, 8)
    
    dct_layer = LinearDCT(8, 'idct', norm='ortho')
    dct_layer.to(x.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)


    x_luma = (x_luma + 128).reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = (x_chroma + 128).reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    x_chroma_orig[:, :, 0::2, 0::2] = x_chroma

    x = torch.cat([x_luma, x_chroma_orig], dim=1)

    x = torch_ycbcr2rgb(x)
    
    # [0, 255] to [-1, 1]
    x = x / 255 * 2 - 1

    return x

def jpeg_encode(x, qf=10):
    # TODO: any quantization
    # Assume x is a batch of size (N x C x H x W)
    # [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255
    n_batch, _, n_size, _ = x.shape
    
    x = torch_rgb2ycbcr(x)
    x_luma, x_chroma = chroma_subsample(x)
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_chroma = unfold(x_chroma).transpose(2, 1)

    x_luma = x_luma.reshape(-1, 8, 8) - 128
    x_chroma = x_chroma.reshape(-1, 8, 8) - 128
    
    dct_layer = LinearDCT(8, 'dct', norm='ortho')
    dct_layer.to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)
    
    x_luma = x_luma.view(-1, 1, 8, 8)
    x_chroma = x_chroma.view(-1, 2, 8, 8)

    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma /= q1.view(1, 8, 8)
    x_chroma /= q2.view(1, 8, 8)
    
    x_luma = x_luma.round()
    x_chroma = x_chroma.round()
    
    x_luma = x_luma.reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = x_chroma.reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)
    
    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)
    
    return [x_luma, x_chroma]



def jpeg_decode(x, qf=10):
    # TODO: any quantization
    # Assume x[0] is a batch of size (N x 1 x H x W) (luma)
    # Assume x[1:] is a batch of size (N x 2 x H/2 x W/2) (chroma)
    x_luma, x_chroma = x
    n_batch, _, n_size, _ = x_luma.shape
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_luma = x_luma.reshape(-1, 1, 8, 8)
    x_chroma = unfold(x_chroma).transpose(2, 1)
    x_chroma = x_chroma.reshape(-1, 2, 8, 8)
    
    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma *= q1.view(1, 8, 8)
    x_chroma *= q2.view(1, 8, 8)
    
    x_luma = x_luma.reshape(-1, 8, 8)
    x_chroma = x_chroma.reshape(-1, 8, 8)
    
    dct_layer = LinearDCT(8, 'idct', norm='ortho')
    dct_layer.to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)
    
    x_luma = (x_luma + 128).reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = (x_chroma + 128).reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    x_chroma_repeated = torch.zeros(n_batch, 2, n_size, n_size, device = x_luma.device)
    x_chroma_repeated[:, :, 0::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 0::2, 1::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 1::2] = x_chroma

    x = torch.cat([x_luma, x_chroma_repeated], dim=1)

    x = torch_ycbcr2rgb(x)
    
    # [0, 255] to [-1, 1]
    x = x / 255 * 2 - 1
    
    return x
