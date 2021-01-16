from collections import namedtuple
from torch import nn
from torch.utils import model_zoo

import lungs_segmentation.unet as Unet

model = namedtuple("model", ["url", "model"])

models = {
    "resnet34": model(
        url="https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/resnet34.pth",
        model=Unet.Resnet(seg_classes=2, backbone_arch="resnet34"),
    ),
    "densenet121": model(
        url="https://github.com/alimbekovKZ/lungs_segmentation/releases/download/1.0.0/densenet121.pth",
        model=Unet.DensenetUnet(seg_classes=2, backbone_arch="densenet121"),
    ),
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(
        models[model_name].url, progress=True, map_location="cpu"
    )  # ["state_dict"]
    model.load_state_dict(state_dict)
    return model
