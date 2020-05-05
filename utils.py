from IPython.core.getipython import get_ipython
import scipy.io
import PIL
import torchvision
import numpy as np
import pathlib

"""
Defining convenient dictionaries for mapping between category ID, LABEL and DESCRIPTION in IMAGE-NET et al.
"""

meta = scipy.io.loadmat('descriptions.mat')
LABEL_TO_DESC = {}
for i in range(1000):
    LABEL_TO_DESC[meta['synsets'][i][0][1][0]] = meta['synsets'][i][0][2][0]

ID_TO_LABEL = dict(enumerate(sorted(LABEL_TO_DESC.keys())))
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
ID_TO_DESC = {k: LABEL_TO_DESC[ID_TO_LABEL[k]] for k in ID_TO_LABEL.keys()}


def find_ids(words_list):
    """
    Helpful function to finding all ids which descriptions contain any word from `words_list`.
    """
    res = []
    for k in ID_TO_DESC:
        val = ID_TO_DESC[k]
        if any(w in val for w in words_list):
            res.append(k)
            print(k, val)
    return res


def load_maps(label_dir, img_num):
    label_dir = pathlib.Path(label_dir)
    return np.load(label_dir / f'res_{str(img_num).zfill(3)}.npz')


def load_and_print_maps(label_dir, img_num):
    maps = load_maps(label_dir, img_num)
    for _, m in maps.items():
        plt.figure()
        plt.imshow(m)
    return maps


def load_image(path):
    """
    Function loads `.png` or `.jpg` image located on `path`.
    Then converts the image to RGB, resize and center cropped it to 224x224 size (fixed size of VGG-16 input).
    """
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    for t in [torchvision.transforms.Resize(256),
              torchvision.transforms.CenterCrop(224)]:
        img = t(img)
    return np.asarray(img).astype('float32') / 255.0


def create_new_cell(contents):
    """
    Function creates new cell in jupyter notebook filled it with `contents`.
    """
    shell = get_ipython()

    payload = dict(
        source='set_next_input',
        text=contents,
        replace=False,
    )
    shell.payload_manager.write_payload(payload, single=False)
