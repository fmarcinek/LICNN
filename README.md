# LICNN
Lateral Inhibition-Inspired Convolutional Neural Network for Visual Attention and Saliency Detection

### How to run the code
In order to use Lateral Inhibition Net simply import it from `LICNN_core.py` file.

Example code:
```
import LICNN_core

net = LICNN_core.LICNN()

# you can use auxiliary function from utils module to load an image in proper format
img = utils.load_image(some_img_path)

# then you can feed network with image to produce output and attention_map
top100, attention_map = net(LICNN_core.to_tensor(img))

# you can also create saliency map using the following classmethod
top100, first_attention_map, saliency_map = LICNN_core.LICNN.create_saliency_map(LICNN_core.to_tensor(img))
```
