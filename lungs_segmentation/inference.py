import torch
import albumentations as A
import numpy as np
import matplotlib
import cv2

COLOR = "white"
matplotlib.rcParams["text.color"] = COLOR
matplotlib.rcParams["axes.labelcolor"] = COLOR
matplotlib.rcParams["xtick.color"] = COLOR
matplotlib.rcParams["ytick.color"] = COLOR


img_size = 512
aug = A.Compose([A.Resize(img_size, img_size, interpolation=1, p=1)], p=1)


def img_with_masks(img, masks, alpha, return_colors=False):
    """
    returns image with masks,
    img - numpy array of image
    masks - list of masks. Maximum 6 masks. only 0 and 1 allowed
    alpha - int transparency [0:1]
    return_colors returns list of names of colors of each mask
    """
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [102, 51, 0],
    ]
    color_names = ["Red", "greed", "BLue", "Yello", "Light", "Brown"]
    img = img - img.min()
    img = img / (img.max() - img.min())
    img *= 255
    img = img.astype(np.uint8)

    c = 0
    for mask in masks:
        mask = np.dstack((mask, mask, mask)) * np.array(colors[c])
        mask = mask.astype(np.uint8)
        img = cv2.addWeighted(mask, alpha, img, 1, 0.0)
        c = c + 1
    if return_colors is False:
        return img
    else:
        return img, color_names[0:len(masks)]


def inference(model, img_path, thresh=0.2):
    model.eval()
    image = cv2.imread(f"{img_path}")
    image = (image - image.min()) / (image.max() - image.min())
    augs = aug(image=image)
    image = augs["image"].transpose((2, 0, 1))
    im = augs["image"]
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image)

    mask = torch.nn.Sigmoid()(model(image.float().cuda()))
    mask = mask[0, :, :, :].cpu().detach().numpy()
    mask = (mask > thresh).astype("uint8")
    return im, mask
