import numpy as np
import matplotlib.pyplot as plt
import PIL
import seaborn as sns

import torch
import torch.nn
import torch.nn.functional as F
import torchvision

import collections
import re
import pathlib

from IPython.core.display import display, HTML

import utils

# seaborn white background
sns.set_style('white')

# expands Jupyter Notebook cells across the entire screen width
display(HTML("<style>.container { width:100% !important; }</style>"))

# matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')

CUDA = torch.cuda.is_available()

"""
Convenient functions to mapping between numpy.array and torch.tensor
"""
def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, **kwargs):
    device = 'cuda' if CUDA else 'cpu'
    return torch.tensor(x, device=device, **kwargs)


class VGGPreprocess(torch.nn.Module):
    """
    PyTorch module that normalizes data for a VGG network
    """
    # These values are taken from http://pytorch.org/docs/master/torchvision/models.html
    RGB_MEANS = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
    RGB_STDS = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]
    
    def forward(self, x):
        """Normalize a single image or a batch of images
        
        Args:
            x: tensor containing and float32 RGB image tensor with 
              dimensions (batch_size x width x heigth x RGB_channels) or 
              (width x heigth x RGB_channels).
        Returns:
            tensor containing a normalized RGB image with shape
              (batch_size x RGB_channels x width x heigth)
        """
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        # x is batch * width * height * channels,
        # make it batch * channels * width * height
        if x.size(3) == 3:
            x = x.permute(0, 3, 1, 2)
        means = self.RGB_MEANS
        stds = self.RGB_STDS
        if x.is_cuda:
            means = means.cuda()
            stds = stds.cuda()
        x = (x - means) / stds
        return x


class VGG(torch.nn.Module):
    """
    Wrapper around a VGG network allowing convenient extraction of layer activations.
    """
    FEATURE_LAYER_NAMES = {
        'vgg16':
            ["conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
             "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
             "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3",
             "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "pool4",
             "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "pool5"],
        'vgg19':
            ["conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1",
             "conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2",
             "conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "conv3_4", "relu3_4", "pool3",
             "conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "conv4_4", "relu4_4", "pool4",
             "conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "conv5_4", "relu5_4", "pool5"]}
    
    def __init__(self, model='vgg16'):
        super(VGG, self).__init__()
        all_models = {'vgg16': torchvision.models.vgg16,
                      'vgg19': torchvision.models.vgg19}   
        vgg = all_models[model](pretrained=True)
        vgg.eval()
        self.preprocess = VGGPreprocess()
        self.features = vgg.features
        self.classifier = vgg.classifier
        self.softmax = torch.nn.Softmax(dim=-1)
        
        self.feature_names = self.FEATURE_LAYER_NAMES[model]
        
        assert len(self.feature_names) == len(self.features)

    def forward(self, x):
        """
        Return pre-softmax unnormalized logits.
        """
        x = self.preprocess(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def logits_and_activations(self, x, layer_names, as_dict=False, suppression_masks={}, save_maps=False, without_last_relu=True):
        x = self.preprocess(x)
        
        needed_layers = set(layer_names)
        if suppression_masks != {}:
            assert all(layer in suppression_masks for layer in needed_layers)
        layer_values = {}
        maps_to_print = collections.defaultdict(list)
        
        for name, layer in zip(self.feature_names, self.features):
            x = layer(x)

            if name in suppression_masks:
                if save_maps:
                    maps_to_print[name].append(to_np(suppression_masks[name].squeeze()))
                    maps_to_print[name].append(to_np(x.squeeze(0).sum(0)))
                
                # applying suppression mask to relu layer
                if not (without_last_relu and name == 'relu5_3'):
                    sup_mask_sized_as_x = suppression_masks[name].expand(-1, x.size()[1], -1, -1)
                    x = x.where(sup_mask_sized_as_x != 0, torch.zeros_like(x))

                if save_maps:
                    maps_to_print[name].append(to_np(x.squeeze(0).sum(0)))
                
            if name in needed_layers:
                layer_values[name] = x
        
        if not as_dict:
            layer_values = [layer_values[n] for n in layer_names]
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x, layer_values, maps_to_print

    def predict(self, x):
        """
        Return predicted class IDs.
        """
        logits = self(x)
        _, prediction = logits.max(1)
        return prediction[0].item()


test_vgg = VGG(model='vgg16')

if CUDA:
    test_vgg.cuda()
    print('VGG is working on cuda')


"""
The following code implements lateral inhibition kernel.
"""

def mex_hat(d):
    grid = (np.mgrid[:d, :d] - d//2) * 1.0
    eucl_grid = (grid**2).sum(0) ** 0.5  # euclidean distances
    eucl_grid /= d  # normalize by LIZ length
    return eucl_grid * np.exp(-eucl_grid)  # mex_hat function values


class LateralInhibition(torch.nn.Module):
    def __init__(self, l=7, a=0.1, b=0.9):
        super().__init__()
        self.len = l
        assert self.len % 2 == 1
        self.a = a
        self.b = b
        self.register_buffer(
            'inhibition_kernel',
            to_tensor(mex_hat(l), dtype=torch.float32).view(1, 1, 1, -1))
    
    def forward(self, x):  # as argument we get max-c map with dimensions 'batch x 1 x n x n'
        assert x.size(1) == 1
        assert x.size(2) == x.size(3)
        len_ = self.len
        pad = len_ // 2
        batches = x.size(0)
        n = x.size(2)
        
        # unfold x to LIZs for each pixel:
        x_unf = F.unfold(x, (len_, len_), padding=(pad, pad))
        # next line is needed for extend tensor size (from 'batch x kernel x n*n' to 'batch x 1 x kernel x n*n'):
        x_unf = x_unf.view(batches, 1, len_*len_, n*n)
        x_unf = x_unf.transpose(2,3)
        # select all middle points in LIZs:
        mid_vals = x.view(x.size(0), 1, n*n, 1)
        
        average_term = torch.exp(-x_unf.mean(3, keepdim=True)).view(batches, 1, n, n)
        
        differential_term = (self.inhibition_kernel * F.relu(x_unf - mid_vals)
                            ).sum(3, keepdim=True).view(batches, 1, n, n)
        
        suppression_mask = self.a * average_term + self.b * differential_term
        assert x.shape == suppression_mask.shape
        suppression_mask_norm = (suppression_mask ** 2).sum() ** 0.5
        suppression_mask /= suppression_mask_norm
        # because all values are non-negative we can do this:
        filter_ = x > suppression_mask
        suppression_mask = x.where(filter_, torch.zeros_like(x))
        return suppression_mask, average_term, differential_term

        # PROBLEM WITH GATING: non-normalized partial computations


"""
The following code implements whole LICNN with the possibility of show suppression masks and 
"""


class LICNN(torch.nn.Module):
    def __init__(self, vgg='vgg16', l=7, a=0.1, b=0.9):
        super().__init__()
        self.vgg = VGG(vgg)
        if CUDA:
            self.vgg.cuda()
        self.lateral_inhibition = LateralInhibition(l, a, b)
        if CUDA:
            self.lateral_inhibition.cuda()
        self.relu_layers = [name for name in self.vgg.feature_names if 'relu' in name]

    def forward(self, img, given_id=None, without_last_relu=True, draw=False, show_masks=False, only_save=False, idx=0):
        self.vgg.zero_grad()  # cleaning for fresh use of LICNN, we have to erase grad due to the automatic accumulation

        # 1 & 2
        logits, acts, _ = self.vgg.logits_and_activations(to_tensor(img), self.relu_layers, as_dict=True,
                                                          without_last_relu=without_last_relu)
        for v in acts.values():
            v.retain_grad()

        _, predictions = torch.topk(logits, 5)
        predictions = predictions[0]
        predicted_id = predictions[0].item()
        if not only_save:
            print('GIVEN_CLASS:', given_id)
            print('FIRST PREDICTION:', utils.ID_TO_DESC[predicted_id])

        loss = F.softmax(logits, dim=-1)[0, given_id if given_id else predictions[idx]] # I got better results applying backpropagation after softmax
        loss.backward()

        # save them for return operation
        top100 = to_np(torch.topk(logits, 100)[1][0])

        # 3
        suppression_masks = {}
        for layer in self.relu_layers:
            gradient = acts[layer].grad
            # creating max-c map
            max_c = gradient.max(1, keepdim=True)[0]
            # max-c map is normalized by L2 norm
            max_c_norm = (max_c ** 2).sum() ** 0.5
            max_c /= max_c_norm
            # generating suppression mask through lateral inhibition
            sup_mask, *_ = self.lateral_inhibition(max_c)
            suppression_masks[layer] = sup_mask

        # 4
        logits, acts, relu_maps = self.vgg.logits_and_activations(
            to_tensor(img), self.relu_layers, as_dict=True, without_last_relu=without_last_relu, suppression_masks=suppression_masks, save_maps=show_masks)

        _, predictions = logits.max(1)
        predicted_id = predictions[0].item()
        if not only_save:
            print('SECOND PREDICTION:', utils.ID_TO_DESC[predicted_id])

        # 5 & 6 & 7
        response_attention_map = self.create_attention_map(img, acts)

        if show_masks and not only_save:
            plt.figure(figsize=(24, 96))
            self.show_maps(relu_maps)

        if draw and not only_save:
            plt.figure(figsize=(24, 96))
            self.print_layers(acts_prev, part=1)
            self.print_layers(acts_now, part=2)

        # show attention maps
            plt.figure()
            plt.imshow(response_attention_map, cmap='plasma')
            plt.colorbar()
            plt.title('norm response attention map')

        return top100, response_attention_map

    def create_attention_map(self, img, acts):
        """
        Function creates response attention map, firstly by summing sum-c maps from all ReLu layers,
          and then normalizing the results by L2 norm.

        """
        img_len = img.shape[1]

        response_based_sum_c_maps = []
        for l in self.relu_layers:
            sum_c_map = acts[l][0].sum(0, keepdim=True).unsqueeze(0)
            map_len = sum_c_map.size(2)
            resized_sum_c_map = F.interpolate(sum_c_map, scale_factor=img_len//map_len, mode='nearest')
            response_based_sum_c_maps.append(to_np(resized_sum_c_map.squeeze(0).squeeze(0)))

        # creating attention map
        response_attention_map = np.array(response_based_sum_c_maps).sum(0)

        # normalizing attention map
        response_attention_map /= (response_attention_map ** 2).sum() ** 0.5

        return response_attention_map

    def show_maps(self, maps):
        """
        Function prints three maps for each ReLu layer: suppression mask, ReLu activation map before lateral inhibition
          and ReLu layer activation after lateral inhibition.
        """
        pnum = 1

        for l in self.relu_layers:
            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            plt.imshow(maps[l][0], cmap='viridis')
            plt.colorbar(shrink=1.0)
            plt.title(f"suppression mask - {l}")

            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            plt.imshow(maps[l][1], cmap='viridis')
            plt.colorbar(shrink=1.0)
            plt.title(f"normal ReLu - {l}")

            plt.subplot(len(self.relu_layers), 3, pnum)
            pnum += 1
            plt.imshow(maps[l][2], cmap='viridis')
            plt.colorbar(shrink=1.0)
            plt.title(f"inhibited ReLu - {l}")

    def print_layers(self, acts, part):
        def increase(pnum):
            return pnum+3+1 if pnum%3 == 0 else pnum+1

        pnum = (part-1)*3 + 1

        for l in self.relu_layers:
            plt.subplot(len(self.relu_layers)*2, 3, pnum)
            pnum = increase(pnum)
            plt.imshow(acts[l].sum(0), cmap='plasma')
            plt.colorbar(shrink=1.0)
            plt.title(f"sum val {l}")

            plt.subplot(len(self.relu_layers)*2, 3, pnum)
            pnum = increase(pnum)
            plt.imshow(acts[l].max(0), cmap='plasma')
            plt.colorbar(shrink=1.0)
            plt.title(f"max val {l}")

            plt.subplot(len(self.relu_layers)*2, 3, pnum)
            pnum = increase(pnum)
            sum_c = acts[l].sum(0)
            norm = (sum_c ** 2).sum()**0.5
            plt.imshow(sum_c / norm, cmap='plasma')
            plt.colorbar(shrink=1.0)
            plt.title(f"norm val {l}")

    @classmethod
    def create_saliency_map(cls, img, show=False):
        ll = LICNN()
        # create attention maps which will create together
        attention_maps = []
        top100 = None
        for i in range(5):
            top100, attention_map = ll(img, idx=i, only_save=True)
            attention_maps.append(attention_map)

        if show:
            for i in range(5):
                plt.figure()
                plt.imshow(attention_maps[i])
                print(utils.ID_TO_DESC[top100[i]])
        
        # create saliency map
        attention_maps = np.array(attention_maps)
        saliency_map = attention_maps.sum(0)

        # normalize saliency map
        norm = (saliency_map ** 2).sum() ** 0.5
        saliency_map /= norm

        first_attention_map = attention_maps[0]

        return top100, first_attention_map, saliency_map
