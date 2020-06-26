# python test_scene_only.py --dataset=places --envtype=home
# python test_scene_only.py --dataset=vpc --hometype=home1 --floortype=data_1

# Prediction for Scene_Only model
#
# by Anwesan Pal

import argparse
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from pathlib import Path

def curdir(thisfile=__file__):
    return Path(__file__).parent

def abspath(rpath, reldir=curdir()):
    return str(reldir / Path(rpath))

def get_semantic_class(img, hometype='home1',
                       floortype='data_0',
                       envtype='home'):
    # th architecture to use
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = abspath('models/{}_best_{}.pth.tar'.format(arch,envtype))

    model = models.__dict__[arch](num_classes=7)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()


    # load the image transformer
    centre_crop = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'categories_places365_{}.txt'.format(envtype)

    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    input_img = V(centre_crop(img).unsqueeze(0))
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    class_name = classes[idx[0]]
    return class_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEDUCE Scene_Only Evaluation')
    parser.add_argument('--img',default='test.png',help='image to test')
    parser.add_argument('--hometype',default='home1',help='home type to test')
    parser.add_argument('--floortype',default='data_0',help='data type to test')
    parser.add_argument('--envtype',default='home',help='home or office type environment')

    args = parser.parse_args()
    kw = vars(args)
    img = Image.open(kw.pop('img'))
    class_name = get_semantic_class(img, **kw)
    print("class : ", class_name)
