import torch
from CLIP import clip
import torchvision.transforms as T
import torchvision.transforms.functional as TF

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CLIP_PERCEPTORS = None

def init_clip(clip_models):
  global CLIP_PERCEPTORS
  if CLIP_PERCEPTORS is None:
    CLIP_PERCEPTORS = [clip.load(model, jit=False)[0].eval().requires_grad_(False).to(DEVICE) for model in clip_models]

def free_clip():
  global CLIP_PERCEPTORS
  CLIP_PERCEPTORS = None


vignette = torch.linspace(-1,1,224).unsqueeze(0).tile(224,1).unsqueeze(0)
vignette = torch.cat((vignette, vignette.rot90(1,(-1,-2))))
vignette = torch.sqrt(vignette[0]**2 + vignette[1]**2)
vignette = vignette.clamp(0,1)**4
vignette = vignette.sub(vignette.min())
vignette = 1 - vignette.div(vignette.max())
vignette = vignette.unsqueeze(0).unsqueeze(0).to(DEVICE)

def noise_vignette(shape):
  return torch.cat([(torch.round(vignette**1 + torch.rand_like(vignette)*0.75).clamp(0,1)) for i in range(shape.shape[0])])


random_lowcrop = torch.nn.Sequential(T.RandomCrop((96)),T.Resize((224,224), T.InterpolationMode.NEAREST),)
random_midcrop = torch.nn.Sequential(T.RandomCrop((144)),T.Resize((224,224), T.InterpolationMode.NEAREST),)
random_maxcrop = torch.nn.Sequential(T.Resize((224,224), T.InterpolationMode.NEAREST),)

def random_crops(img):
  return torch.cat([random_lowcrop(img), random_midcrop(img), random_maxcrop(img) * noise_vignette(img)])
  #return torch.cat([random_336crop(img), random_448crop(img)])


CLIP_Normalization = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

def Random_Normalization(img, prob):
    dice_roll = torch.rand((1,)).item()
    if dice_roll >= prob:
        return CLIP_Normalization(img)
    else:
        return img

def random_grayscale(img, prob):
    dice_roll0 = torch.rand((1,)).item()
    dice_roll1 = torch.rand((1,)).item()
    if dice_roll0 >= prob:
        return torch.lerp(img, TF.rgb_to_grayscale(img), dice_roll1)
    else:
        return img