import torch
import torchvision.transforms.functional as TF
from PIL import Image


def scale_image(img):
    img = TF.to_pil_image(img)
    new_width = 244  # set the desired width of the resized image
    width, height = img.size
    # calculate the corresponding height
    new_height = int(height * new_width / width)
    resized_img = TF.resize(img, (new_height, new_width))  # resize the image
    resized_img = TF.center_crop(resized_img, (224, 224))  # crop the image

    img_tensor = TF.to_tensor(resized_img)

    return img_tensor
