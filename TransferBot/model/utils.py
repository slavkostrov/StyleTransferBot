import os

import torch
from PIL import Image
from torchvision import transforms





def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


unloader = transforms.ToPILImage()


def get_pil_image(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return unloader(image)


def save_image_epoch(tensor, path, num):
    """Save a single image."""
    image = get_pil_image(tensor)
    image.save(os.path.join(path, "out_" + str(num) + '.jpg'))


def normalize(img):
    # normalize using imagenet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda()
    return (img - mean) / std
