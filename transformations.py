import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter


def erase_image_background(img, atten_map, show=False):
    if show:
        plt.figure()
        plt.imshow(img)

    # threshold calculation
    max_, min_ = np.max(atten_map), np.min(atten_map)
    threshold = (max_ - min_) / 5.0

    # we have to expand attention_map to img shape
    enlarged_atten_map = atten_map[:, :, np.newaxis]
    enlarged_atten_map = np.concatenate(tuple([enlarged_atten_map]) * 3, axis=2)

    # attention erased image calculation
    attention_erased_img = np.where(enlarged_atten_map > threshold, img, 0.0)

    if show:
        plt.figure()
        plt.imshow(attention_erased_img)

    return attention_erased_img


def blur_image_background(img, atten_map, r=2, show=False):
    if show:
        plt.figure()
        plt.imshow(img)

    # blurring the whole image
    im = Image.fromarray(np.uint8(img * 255))
    blurred_img = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=r)), dtype=np.float32) / 255.0

    # threshold calculation
    max_, min_ = np.max(atten_map), np.min(atten_map)
    threshold = (max_ - min_) / 5.0

    # we have to expand attention_map to img shape
    enlarged_atten_map = atten_map[:, :, np.newaxis]
    enlarged_atten_map = np.concatenate(tuple([enlarged_atten_map]) * 3, axis=2)

    # attention blurred image calculation
    attention_blurred_img = np.where(enlarged_atten_map > threshold, img, blurred_img)

    if show:
        plt.figure()
        plt.imshow(attention_blurred_img)

    return attention_blurred_img


def blur_image_saliency_object(img, atten_map, r=2, show=False):
    if show:
        plt.figure()
        plt.imshow(img)

    # blurring the whole image
    im = Image.fromarray(np.uint8(img * 255))
    blurred_img = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=r)), dtype=np.float32) / 255.0

    # threshold calculation
    max_, min_ = np.max(atten_map), np.min(atten_map)
    threshold = (max_ - min_) / 5.0

    # we have to expand attention_map to img shape
    enlarged_atten_map = atten_map[:, :, np.newaxis]
    enlarged_atten_map = np.concatenate(tuple([enlarged_atten_map]) * 3, axis=2)

    # attention blurred image calculation
    attention_blurred_img = np.where(enlarged_atten_map > threshold, blurred_img, img)

    if show:
        plt.figure()
        plt.imshow(attention_blurred_img)

    return attention_blurred_img
