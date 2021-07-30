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
from torch.utils.mobile_optimizer import optimize_for_mobile
from matplotlib import pyplot as plt
from ofa.model_zoo import ofa_specialized
from ofa.model_zoo import ofa_net
from ofa.utils import download_url
from ofa.elastic_nn.utils import set_running_statistics
#from subnetfinder import EvolutionFinder
from imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
print('The OFA Network is ready.')
#cuda_available = torch.cuda.is_available()
#if cuda_available:
#    torch.backends.cudnn.enabled = True
#    torch.backends.cudnn.benchmark = True
#    torch.cuda.manual_seed(random_seed)
#    print('Using GPU.')
#else:
#    print('Using CPU.')
def prepare_subnet(ofa_net, data_loader,bn_list,net_config):
    assert 'ks' in net_config and 'd' in net_config and 'e' in net_config
    assert len(net_config['ks']) == 20 and len(net_config['e']) == 20 and len(net_config['d']) == 5
    ofa_net.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    subnet = ofa_net.get_active_subnet()
    bn_mean = bn_list[0]
    bn_var = bn_list[1]
    calib_bn(subnet,data_loader,bn_mean,bn_var)
    
    #import torchvision
    #subnet = torchvision.models.mobilenet_v3_small(pretrained=True)
    #When I run torchvision's model. it can run successfully.
    #fuse model
    fuse_model(subnet)  
    #quantize model
    subnet.qconfig = torch.quantization.default_qconfig
    subnet.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'
    torch.quantization.prepare(subnet, inplace=True)
    # Calibrate
    torch.quantization.convert(subnet, inplace=True)
    print(subnet)
    print(type(subnet))
    #optimize
    script_subnet = torch.jit.script(subnet)
    script_subnet_optimized = optimize_for_mobile(script_subnet)
    return script_subnet_optimized

def calib_bn(net,data_loader,bn_mean,bn_var):
    print('Creating dataloader for resetting BN running statistics...')
    set_running_statistics(net, data_loader,bn_mean,bn_var)

def fuse_model(model):
    modules_names = [m for m in model.named_modules()]
    modules_list = [m for m in model.modules()]
    for ind, m in enumerate(modules_list):
        if type(m) == nn.Conv2d and type(modules_list[ind+1]) == nn.BatchNorm2d and type(modules_list[ind+2]) == nn.ReLU:
            '''
            print("Find ConvBNReLu: ", modules_names[ind][0], '-->', 
                modules_names[ind+1][0], '-->', modules_names[ind+2][0])'''
            torch.quantization.fuse_modules(model, [modules_names[ind][0],
                modules_names[ind+1][0], modules_names[ind+2][0]], inplace=True)
        elif type(m) == nn.Conv2d and type(modules_list[ind+1]) == nn.BatchNorm2d:
            '''
            print("Find ConvBN: ", modules_names[ind][0], '-->', modules_names[ind+1][0])'''
            torch.quantization.fuse_modules(model, 
               [modules_names[ind][0], modules_names[ind+1][0]], inplace=True)
        elif type(m) == nn.Conv2d and type(modules_list[ind+1]) == nn.ReLU:
            '''
            print("Find ConvReLU: ", modules_names[ind][0], '-->', modules_names[ind+1][0])'''
            torch.quantization.fuse_modules(model,
               [modules_names[ind][0], modules_names[ind+1][0]], inplace=True)


imagenet_data_path = '/home/yuantian/DynamicOFA/data/'
#imagenet_data_path = '/home/lewislou/data/imagenet'
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

optimal_networks = []
for i in range(1):  
    net_config = mylist['configs'][i]
    bn_list = mylist['bn_lists'][i]
    optimal_networks.append(prepare_subnet(ofa_network,
	data_loader,bn_list[0],net_config))

top1s = []
latency_list = []
init_latencys = []
print("The len of sub-networks:, ", len(optimal_networks))
for i in range(len(optimal_networks)):
	print("The network: ", i)
	top1,latency= evaluate_ofa_subnet(optimal_networks[i],
		imagenet_data_path, mylist['configs'][i], data_loader, batch_size=16)
	print('top1 accuracy is',top1)
	print('init_latency and inference latency',latency)

