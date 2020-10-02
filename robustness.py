# -*- coding: utf-8 -*-

import argparse
import os
import time
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import pandas as pd

torch.nn.Module.dump_patches = True

# /////////////// Model Setup ///////////////
torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
test_bs = 100

models = {
    2:"result/False_-1_False_False_cifar10_300_64_2_False_resnet18_297467904.0_0_1000.0_10.0_0.0005.pt",
    1:"result/False_-1_False_False_cifar10_300_64_1_False_resnet18_555417600.0_633833_1000.0_10.0_0.0005.pt",
    4:"result/False_-1_False_False_cifar10_300_64_4_False_resnet18_168493056.0_155487_1000.0_10.0_0.0005.pt",
    8:"result/False_-1_False_False_cifar10_300_64_8_False_resnet18_104005632.0_262394_1000.0_10.0_0.0005.pt",
    16:"result/False_-1_False_False_cifar10_300_64_16_False_resnet18_71761920.0_488910_1000.0_10.0_0.0005.pt",
    32:"result/False_-1_False_False_cifar10_300_64_32_False_resnet18_55640064.0_279824_1000.0_10.0_0.0005.pt",
    64: "result/False_-1_False_False_cifar10_300_64_64_False_resnet18_47579136.0_13181_1000.0_10.0_0.0005.pt",
}

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

prefetch = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
clean_loader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False, num_workers=prefetch,pin_memory=True)

dataframeStarted = None
dataframe = None

for l,model_name in models.items():
    print("L={}".format(l))
    checkpoint = torch.load(model_name)
    net = checkpoint
    net.output_relus = True

    net.cuda()
    #net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True


    net.eval()

    print('Model Loaded')

    # /////////////// Data Loader ///////////////

    transform = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    correct = 0
    for batch_idx, (data, target) in enumerate(clean_loader):
        all_data = list()
        for _input in data:
            all_data.append(transform(_input).view(1,*_input.size()))
        all_data = torch.cat(all_data)
        data = all_data.cuda()

        output = net.temp_forward(data,l,-1,0)

        pred = output.max(1)[1]
        correct += pred.eq(target.cuda()).sum()

    clean_error = 1 - correct.float() / len(clean_loader.dataset)
    clean_error = clean_error.cpu().numpy()
    print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


    # /////////////// Further Setup ///////////////

    def auc(errs):  # area under the distortion-error curve
        area = 0
        for i in range(1, len(errs)):
            area += (errs[i] + errs[i - 1]) / 2
        area /= len(errs) - 1
        return area


    def show_performance(distortion_name):
        with torch.no_grad():
            errs = []
            labels = np.load("data/labels.npy")
            dataset = np.load("data/{}.npy".format(distortion_name))
            dataset = np.transpose(dataset,[0,3,1,2])

            for severity in range(0, 5):
                torch_data = torch.FloatTensor(dataset[10000*severity:10000*(severity+1)])
                torch_labels = torch.LongTensor(labels[10000*severity:10000*(severity+1)])
                test = torch.utils.data.TensorDataset(torch_data, torch_labels)
                distorted_dataset_loader = torch.utils.data.DataLoader(test, batch_size=test_bs, shuffle=False,num_workers=prefetch,pin_memory=True)



                correct = 0
                for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
                    all_data = list()
                    for _input in data:
                        all_data.append(transform(_input/255).view(1,*_input.size()))
                    all_data = torch.cat(all_data)
                    data = all_data.cuda()

                    output = net.temp_forward(data,l,-1,0)

                    pred = output.max(1)[1]
                    correct += pred.eq(target.cuda()).sum()
                percentage = correct.float() / 10000
                errs.append( (1 - percentage ).item())

            print('\n=Average', tuple(errs))
            return errs


    # /////////////// End Further Setup ///////////////


    # /////////////// Display Results ///////////////
    import collections

    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]

    error_rates = list()
    result_dict = dict(l=l,model_name=model_name,clean_error=clean_error)
    for distortion_name in distortions:
        rate = show_performance(distortion_name)
        result_dict[distortion_name] = np.mean(rate)
        print('Distortion: {:15s}  | Error (%): {:.2f}'.format(distortion_name, 100 * np.mean(rate)))
        error_rates.append(np.mean(rate))
    if not dataframeStarted:
        dataframe = pd.DataFrame(result_dict,index=[0])
        dataframeStarted = True
    else:
        dataframe = pd.concat([dataframe,pd.DataFrame(result_dict,index=[0])])
    dataframe.to_csv("robustness.csv")

    print('Mean Error (%): {:.2f}'.format(100 * np.mean(error_rates)))