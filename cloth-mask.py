import os
from PIL import Image
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from networks.u2net import U2NET
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = '/content/inputs/test/cloth'
result_dir = '/content/inputs/test/cloth-mask'
checkpoint_path = 'cloth_segm_u2net_latest.pth'

def load_checkpoint_mgpu(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return model  # Return the model even if loading fails
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model

class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=[mean] * 3, std=[std] * 3)

    def __call__(self, image_tensor):
        return self.normalize(image_tensor)


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        palette[j * 3:(j + 1) * 3] = [255, 255, 255]  # Set all to white for simplicity
    return palette


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

print("Initializing model...")
try:
    net = U2NET(in_ch=3, out_ch=4)
except Exception as e:
    print(f"Error initializing model: {e}")
    net = None  # Ensure net is None if initialization fails

net = load_checkpoint_mgpu(net, checkpoint_path)

if net is None:
    raise ValueError("Model initialization failed after loading checkpoint.")

net = net.to(device)
net = net.eval()

palette = get_palette(4)

images_list = sorted(os.listdir(image_dir))
for image_name in images_list:
    try:
        img = Image.open(os.path.join(image_dir, image_name)).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_name}: {e}")
        continue
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    output_img = Image.fromarray(output_arr.astype('uint8'), mode='L')
    output_img = output_img.resize(img_size, Image.BICUBIC)
    
    output_img.putpalette(palette)
    output_img = output_img.convert('L')
    output_img.save(os.path.join(result_dir, image_name[:-4]+'.jpg'))