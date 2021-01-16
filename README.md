# lungs_segmentation
Automated lung segmentation in chest-x ray 

![https://habrastorage.org/webt/vk/jv/8r/vkjv8rjd04f1oicbczq5hyadhv0.png](https://habrastorage.org/webt/vk/jv/8r/vkjv8rjd04f1oicbczq5hyadhv0.png)

## Installation

`pip install lungs-segmentation`

### Example inference

Jupyter notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hNNl-tHipIBGmtexU2BH8qC80-QxXFZt?usp=sharing)

### WebApp

Comming soon ...

### Models weights

| model | best dice | Mb |
|-------|-----------|----|
|   resnet34    | 0.9657          |  103.4  |
|   densenet121    |  0.9655         |   62.8 |

### Usage

Code example for resnet34:

```
from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference

model = create_model("resnet34")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

plt.figure(figsize=(20,40))
plt.subplot(1,1,1)
image, mask = inference.inference(model,'739px-Chest_Xray_PA_3-8-2010.png', 0.2)
plt.imshow(inference.img_with_masks( image, [mask[0], mask[1]], alpha = 0.1))
```


### Results on data from the Internet

#### resnet34

![https://habrastorage.org/webt/e3/mb/kc/e3mbkcxsmos6q4jlw5-tybudzji.png](https://habrastorage.org/webt/e3/mb/kc/e3mbkcxsmos6q4jlw5-tybudzji.png)

#### densenet121

![https://habrastorage.org/webt/ef/01/zo/ef01zo2g2qgsux8ses4keg4g8is.png](https://habrastorage.org/webt/ef/01/zo/ef01zo2g2qgsux8ses4keg4g8is.png)

### Authors

![Renat Alimbekov](https://alimbekov.com)

![Ivan Vassilenko](https://www.linkedin.com/in/ivannvassilenko/)

![Abylaikhan Turlassov](https://www.linkedin.com/in/abylaikhan-turlassov-2727b2196/)
