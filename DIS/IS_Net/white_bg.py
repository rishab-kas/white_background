import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import os

import requests
import matplotlib.pyplot as plt
from io import BytesIO

# project imports
from .data_loader_cache import normalize, im_reader, im_preprocess
from .models import *

class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image


transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])

def load_image(im_path, hypar):
    if im_path.startswith("http"):
        im_path = BytesIO(requests.get(im_path).content)

    im = im_reader(im_path)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[:, :, :3]
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0) # make a batch of image, shape


def build_model(hypar,device):
    net = hypar["model"]#GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if(hypar["restore_model"]!=""):
        net.load_state_dict(torch.load(hypar["model_path"]+"/"+hypar["restore_model"],map_location=device))
        net.to(device)
    net.eval()
    return net


def predict(net,  inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if(hypar["model_digit"]=="full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)


    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable

    ds_val = net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[0][0],shapes_val[0][1]),mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need

def predictmask(local_image_path,net,hypar,device):

    with open(local_image_path, 'rb') as file:
        image_content = file.read()

    # Create BytesIO object with the image content
    image_bytes = BytesIO(image_content)

    image_tensor, orig_size = load_image(local_image_path, hypar)
    mask = predict(net,image_tensor,orig_size, hypar, device)


    orig = np.array(Image.open(image_bytes))
    if orig.shape[-1] == 4:
        orig = orig[..., :3]
    return mask, orig

def mask_to_img(mask,orig):
  # Convert the mask to a 3-channel mask for compatibility with the original image
  mask_rgb = np.stack((mask, mask, mask), axis=-1)

  # Apply the mask to the original image
  masked_image = np.where(mask_rgb, orig, 255)
  return masked_image

def define_parameters():
  hypar = {} # paramters for inferencing

  hypar["model_path"] ="./saved_new" ## load trained weights from this path
  

  # Step 1: Get the directory of the current script (inference code)
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # Step 2: Construct the absolute path to the model weights
  # Assuming the weights are located in a subdirectory named 'models' relative to the script
  hypar["model_path"] = os.path.join(script_dir, 'saved_models')

  # hypar["model_path"] ="./saved_models" ## load trained weights from this path

  hypar["restore_model"] = "isnet.pth" ## name of the to-be-loaded weights
  hypar["restore_model"] = "gpu_itr_25800_traLoss_0.0751_traTarLoss_0.0052_valLoss_0.0579_valTarLoss_0.005_maxF1_0.999_mae_0.0011_time_0.058883.pth" ## name of the to-be-loaded weights
  # hypar["restore_model"] = "gpu_itr_5600_traLoss_0.0885_traTarLoss_0.0058_valLoss_1.4584_valTarLoss_0.217_maxF1_0.9709_mae_0.029_time_0.051693.pth" ## name of the to-be-loaded weights

  hypar["interm_sup"] = False ## indicate if activate intermediate feature supervision

  ##  choose floating point accuracy --
  hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
  hypar["seed"] = 0

  hypar["cache_size"] = [1024, 1024] ## cached input spatial resolution, can be configured into different size

  ## data augmentation parameters ---
  hypar["input_size"] = [1024, 1024] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
  hypar["crop_size"] = [1024, 1024] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation

  hypar["model"] = ISNetDIS()
  return hypar

def white_bg_generate(input_path,output_path):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  hypar = define_parameters()
  net = build_model(hypar, device)
  mask_blackwhite,original = predictmask(input_path,net,hypar,device)
  final = mask_to_img(mask_blackwhite,original)
  plt.imsave(output_path,final)
