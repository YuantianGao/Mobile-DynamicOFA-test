# In[ ]:


import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import torchvision
import random
import math
import copy
import pickle
import ofa
import torch.quantization

import torch.backends
from torch.utils.mobile_optimizer import optimize_for_mobile
from matplotlib import pyplot
from quantize import QuantizedModel

from torchvision import datasets, transforms as T
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)

torch.backends.cudnn.enabled = False

torch.backends.cudnn.benchmark = False
#cuda_available = torch.cuda.is_available()
#if cuda_available:
#    torch.backends.cudnn.enabled = True
#    torch.backends.cudnn.benchmark = True
#    torch.cuda.manual_seed(random_seed)
#    print('Using GPU.')
#else:
#    print('Using CPU.')


def fuse_model(model):
    modules_names = [m for m in model.named_modules()]
    modules_list = [m for m in model.modules()]
    for ind, m in enumerate(modules_list):
        if type(m) == nn.Conv2d and type(modules_list[ind+1]) == nn.BatchNorm2d and type(modules_list[ind+2]) == nn.ReLU:
            torch.quantization.fuse_modules(model, [modules_names[ind][0], modules_names[ind+1][0], modules_names[ind+2][0]], inplace=True)
        elif type(m) == nn.Conv2d and type(modules_list[ind+1]) == nn.BatchNorm2d:
            torch.quantization.fuse_modules(model, [modules_names[ind][0], modules_names[ind+1][0]], inplace=True)
        elif type(m) == nn.Conv2d and type(modules_list[ind+1]) == nn.ReLU:
            torch.quantization.fuse_modules(model, [modules_names[ind][0], modules_names[ind+1][0]], inplace=True)


def MobileModel(model):
    #import torchvision
    #model = torchvision.models.mobilenet_v3_small(pretrained=True)
    model = QuantizedModel(model)
    #When I run torchvision's model. it can run successfully.
    #fuse model
    
    fuse_model(model)
    #quantize model
    model.qconfig = torch.quantization.default_qconfig
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'
    torch.quantization.prepare(model, inplace=True)
    # Calibrate
    #print('Calibrating........................')
    evaluate_ofa_subnet(subnet,
        imagenet_data_path, net_config, data_loader, batch_size=16)
    #Convert model
    torch.quantization.convert(model, inplace=True)
    #optimize
    #print(type(subnet))
    script_subnet = torch.jit.script(model)
    #print(script_subnet)
    script_subnet_optimized = optimize_for_mobile(script_subnet)
    return script_subnet_optimized


    
    
imagenet_data_path = '/home/yuantian/DynamicOFA/data/'
def build_val_transform(size):
    return transforms.Compose([
        transforms.Resize(int(math.ceil(size / 0.875))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        root=os.path.join(imagenet_data_path, 'validations'),
        transform=build_val_transform(224)
        ),
    batch_size=1,  # test batch size
    shuffle=False,
    #num_workers=16,  # number of workers for the data loader
    pin_memory=False,
    drop_last=False,
    )

print('The ImageNet dataloader is ready.')
data_loader.dataset
torch.set_num_threads(1)
mylist=[]
file=open(r"nets_7.pickle","rb")
mylist=pickle.load(file)

for i in range(7):
    model = torch.load('/home/yuantian/DynamicOFA/model/model_' + str(i) + '.pth')
	print("The network: ", i)
	model = MobileModel(model)
	torch.jit.save(model, '/home/yuantian/DynamicOFA/model/script_model_' + str(i) + '.pth')
	
	

for i in range(7):
    script_model = torch.load('/home/yuantian/DynamicOFA/model/script_model_' + str(i) + '.pth')
	print("The network: ", i)
	top1,latency= evaluate_ofa_subnet(script_model,
		imagenet_data_path, mylist['configs'][i], data_loader, batch_size=16)
	print('top1 accuracy is',top1)
	print('init_latency and inference latency',latency)

