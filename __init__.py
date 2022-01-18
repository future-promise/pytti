import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn
from scipy import ndimage
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def named_rearrange(tensor, axes, new_positions):
  """
  Permute and unsqueeze tensor to match target dimensional arrangement
  tensor:        (Tensor) input
  axes:          (string tuple) names of dimensions in tensor
  new_positions: (string tuple) names of dimensions in result
                 optionally including new names which will be unsqueezed into singleton dimensions
  """
  #this probably makes it slower honestly
  if axes == new_positions:
    return tensor
  #list to dictionary pseudoinverse
  axes = {k:v for v,k in enumerate(axes)}
  #squeeze axes that need to be gone
  missing_axes = [d for d in axes if d not in new_positions]
  for d in missing_axes:
    dim = axes[d]
    if tensor.shape[dim] != 1:
      raise ValueError(f"Can't convert tensor of shape {tensor.shape} due to non-singelton axis {d} (dim {dim})")
    tensor = tensor.squeeze(axes[d])
    del axes[d]
    axes.update({k:v-1 for k,v in axes.items() if v > dim})
  #add singleton dimensions for missing axes
  extra_axes = [d for d in new_positions if d not in axes]
  for d in extra_axes:
    tensor = tensor.unsqueeze(-1)
    axes[d] = tensor.dim()-1
  #permute to match output
  permutation = [axes[d] for d in new_positions]
  return tensor.permute(*permutation)

def format_input(tensor, source, dest):
  return named_rearrange(tensor, source.output_axes, dest.input_axes)

def pad_tensor(tensor, target_len):
  l = tensor.shape[-1]
  if l >= target_len:
    return tensor
  return F.pad(tensor, (0,target_len-l))

def cat_with_pad(tensors):
  max_size = max(t.shape[-1] for t in tensors)
  return torch.cat([pad_tensor(t, max_size) for t in tensors])

def format_module(module, dest, *args, **kwargs):
  return format_input(module(*args, **kwargs), module, dest)

class ReplaceGrad(torch.autograd.Function):
  """
  returns x_forward during forward pass, but evaluates derivates as though
  x_backward was retruned instead.
  """
  @staticmethod
  def forward(ctx, x_forward, x_backward):
    ctx.shape = x_backward.shape
    return x_forward
  @staticmethod
  def backward(ctx, grad_in):
    return None, grad_in.sum_to_size(ctx.shape)
replace_grad = ReplaceGrad.apply

class ClampWithGrad(torch.autograd.Function):
  """
  clamp function
  """
  @staticmethod
  def forward(ctx, input, min, max):
    ctx.min = min
    ctx.max = max
    ctx.save_for_backward(input)
    return input.clamp(min, max)
  @staticmethod
  def backward(ctx, grad_in):
    input, = ctx.saved_tensors
    return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
clamp_with_grad = ClampWithGrad.apply

def clamp_grad(input, min, max):
  return replace_grad(input.clamp(min,max), input)

def tv_loss(input):
  """L2 total variation loss, as in Mahendran et al."""
  input = F.pad(input, (0, 1, 0, 1), 'replicate')
  x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
  y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
  return (x_diff**2 + y_diff**2).mean([1, 2, 3])

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])


def saturation_loss(input, saturation_weight = 1):
  # based on the old "percepted colourfulness" heuristic from Hasler and Süsstrunk’s 2003 paper
  # https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images
  _pixels = input.permute(0, 2, 3, 1).reshape(-1, 3)
  rg = _pixels[:, 0]-_pixels[:, 1]
  yb = 0.5*(_pixels[:, 0]+_pixels[:, 1])-_pixels[:, 2]
  rg_std, rg_mean = torch.std_mean(rg)
  yb_std, yb_mean = torch.std_mean(yb)
  std_rggb = torch.sqrt(rg_std**2 + yb_std**2)
  mean_rggb = torch.sqrt(rg_mean**2 + yb_mean**2)
  colorfullness = std_rggb+.3*mean_rggb
  return -colorfullness*saturation_weight/10.0

def symmetry_loss(input, weight = 1):
  mseloss = nn.MSELoss()
  cur_loss = mseloss(input, torch.flip(input,[3])) 
  return cur_loss * weight


def sobel_filters(img):
    Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    Ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)

    Ix = nn.functional.conv2d(img, Kx,stride=1, padding=0)
    Iy = nn.functional.conv2d(img, Ky,stride=1, padding=0)

    G = torch.hypot(Ix, Iy)
    G = G / torch.amax(G) * 255
    
    return G

def contrast_loss_edge(input):
  gray = transforms.Grayscale()
  gray_sobel_input = gray(input)
  sobel_mask = sobel_filters(gray_sobel_input)
  print('gray input', gray_sobel_input.shape, sobel_mask.shape, torch.amin(sobel_mask), torch.amax(sobel_mask))
  #sobel_mask_clamped = mask / 255
  #sobel_mask_converted = 1.25 + (1 * sobel_mask_clamped)
  #adjusted = (sobel_mask_converted * (input - 0.5)) + 0.5
  #return torch.clamp(adjusted, min=0, max=1)
  return

def contrast_loss(input, weight = 1, contrast_diff_weight = 1.25, brightness = 10):
  contrasted = (contrast_diff_weight * (input - 0.5)) + 0.5
  contrasted = torch.clamp(contrasted, min=0, max=1)
  #print('contrast input', input)
  #print('contrast output', contrasted)
  mseloss = nn.MSELoss()
  cur_loss = mseloss(input, contrasted)
  return cur_loss * weight * 10.0

def contrast_loss_grayscale(input, weight = 1, contrast_diff_weight = 1.25, brightness = 10):
  gray = transforms.Grayscale()
  gray_input = gray(input)

  contrasted = (contrast_diff_weight * (gray_input - 0.5)) + 0.5
  contrasted = torch.clamp(contrasted, min=0, max=1)
  #print('contrast input', input)
  #print('contrast output', contrasted)
  mseloss = nn.MSELoss()
  cur_loss = mseloss(gray_input, contrasted)
  return cur_loss * weight * 10.0