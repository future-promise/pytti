from pytti import *
from pytti.Image import DifferentiableImage
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from PIL import Image

class TestImage(DifferentiableImage):
  """
  differentiable image format for pixel art images
  """
  def __init__(self, width, height, scale, pallet_size, n_pallets = 2, device=DEVICE):
    super().__init__(width*scale, height*scale)
    self.pallet_inertia = 1#math.pow(width*height*(n_pallets+1),1/3)/math.sqrt(n_pallets*pallet_size*3) 
    pallet = torch.linspace(0,self.pallet_inertia,pallet_size).view(pallet_size,1,1).repeat(1,n_pallets,3)
    #pallet.set_(torch.rand_like(pallet)*self.pallet_inertia)
    self.pallet = nn.Parameter(pallet.to(device))

    self.pallet_size = pallet_size
    self.n_pallets = n_pallets
    self.value  = nn.Parameter(torch.zeros(height,width).to(device))
    self.tensor = nn.Parameter(torch.zeros(n_pallets, height, width).to(device))
    self.output_axes = ('n', 's', 'y', 'x')
    self.scale = scale
    print("init pixel image")

