from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
import mix
import random
import os
import math
from datasets import get_data_loaders
from models import *

import time
start_time = time.time()
start_epoch = time.time()

def quadratic_weight_decay(parameters):
    return 0.5*torch.pow(parameters,2).sum()

def swd(parameters, a, width):
    if a < 0:
        return quadratic_weight_decay(parameters)
    else:
        return 0.5 * a * torch.pow((torch.abs(parameters) < width).float() * parameters, 2).sum() + 0.5 * torch.pow((torch.abs(parameters) >= width).float() * parameters, 2).sum()

def weight_decay(model, a, width):
    total = 0
    for parameters in model.parameters():
        total += swd(parameters, a, width)
    return total

def duration(eta):
    eta = int(eta)
    hours = eta // 3600
    mins = (eta % 3600) // 60
    secs = (eta % 60)
    return "{:3d}h{:02d}m{:02d}s".format(hours, mins, secs)

def train(model, epoch, args, device, train_loader, optimizer, masks = None):
    global start_epoch
    start_epoch = time.time()
    model.train()
    accuracy=0
    total_loss = 0
    last_tick = time.time() - 2
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if args.mixup:
            data, target_a, target_b, lam = mix.mixup_data(data, target, use_cuda = torch.cuda.is_available())
        if args.cutmix:
            data, target_a, target_b, lam = mix.cutmix_data(data, target, use_cuda = torch.cuda.is_available())
        if args.half:
            data = data.half()
        optimizer.zero_grad()
        output = model(data)
        if args.mixup or args.cutmix:
            loss = mix.mix_criterion(torch.nn.CrossEntropyLoss(), output, target_a, target_b, lam)
        else:
            loss = F.cross_entropy(output, target)
        total_loss += loss.item()

        a = args.a

        values = None
        for parameters in model.parameters():
            if values == None:
                if args.half:
                    values = torch.abs(parameters.data.view(-1).half())
                else:
                    values = torch.abs(parameters.data.view(-1))
            else:
                if args.half:
                    values = torch.cat([values,torch.abs(parameters.data.view(-1).half())], dim=0)
                else:
                    values = torch.cat([values,torch.abs(parameters.data.view(-1))], dim=0)
        values = torch.sort(values)[0]
        target_th = int((1-args.target) * 100000)
        th = values[((target_th * values.shape[0]) // 100000)]
        width = th
        
        if masks == None:
            loss += args.wd * weight_decay(model, a, width)

        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item()

        if not masks == None:
            with torch.no_grad():
                for i,parameter in enumerate(model.parameters()):
                    parameter.data = parameter.data * masks[i]
        
        if time.time() - last_tick >= 0.2 or batch_idx == len(train_loader) - 1:
            last_tick = time.time()
            delay = time.time() - start_time
            done = max(1,(epoch * len(train_loader) + batch_idx))
            eta = (args.epochs * len(train_loader) / done * delay) - delay
            print("\r{:3d} \033[1;32m{:5f} ({:.4f}) \033[1;37m({:3d}%) ({:s}, {:s}) {:.5f}".format(epoch+1,total_loss / (batch_idx+1), 1 - (accuracy / ((1+batch_idx) * args.batch_size)), int(100 * (1+batch_idx) / len(train_loader)),duration(eta), duration(time.time() - start_epoch), th),end='')
    train_acc = accuracy / ((1+batch_idx) * args.batch_size)
    return { "train_loss" : total_loss / (1+batch_idx), "train_acc" : train_acc}

def test(model, args, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.half:
                data = data.half()
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct / len(test_loader.dataset)    
    return { "test_loss" : test_loss, "test_acc" : test_acc }

def display_progress(epoch, train_data, test_data):
    scale, color = 1, "\033[1;34m"
    print("\r                                                                 \r",end='')
    print("\r{:3d} \033[1;32m{:5f} ({:.4f}) \033[1;34m{:5f} ({:.4f}) \033[1;37m".format(epoch + 1, train_data["train_loss"], 1 - train_data["train_acc"], test_data["test_loss"], 1 - test_data["test_acc"]), end ='')
    test_acc = int(8 * scale * 100 * (1 - test_data["test_acc"]))
    print(color, end='')
    for i in range(test_acc // 8):
        print('\u2588', end ='')
    last_symbol = ['\u258F','\u258E','\u258D','\u258C','\u258B','\u258A','\u2589','\u2588']
    print(last_symbol[test_acc % 8] + "\033[1;37m")



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SWD')

    # Mandatory Arguments
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help="The dataset to consider")

    parser.add_argument('--lr', type=float, default=.1, metavar='LR',
                        help='learning rate (default: .1) (negative -> Adam)')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train')

    parser.add_argument('--ft-epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--seed", type=int, default=random.randint(0,1000000),
                        help = "random seed to initialize.")

    parser.add_argument("--cutout", action="store_true",
                        help = "perform cutout")

    parser.add_argument("--mixup", action="store_true",
                        help = "perform mixup")

    parser.add_argument("--cutmix", action="store_true",
                        help = "perform cutmix")

    parser.add_argument("--auto-augment", action="store_true",
                        help = "perform auto_augment")

    parser.add_argument('--feature-maps', type=int, default=64,
                        help='Total feature_maps')

    parser.add_argument('--wd', default = "5e-4", type=float,
                        help='Weight decay')

    parser.add_argument('--a', default = "-1", type=float,
                        help='Parameter a')

    parser.add_argument('--target', default="0.01", type=float,
                        help="Target kept ratio")

    parser.add_argument('--half', action='store_true',
                        help='Half precision')

    ## ----------------------------------------------------------------------------------------
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)

    train_loader, test_loader, metadata = get_data_loaders(args)
    model = ResNet20(args).to(device)

    if args.half:
        model.half()  # convert to half precision
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.float()

    n_params = torch.sum(torch.LongTensor([elt.numel() for elt in model.parameters()])).item()
    print(str(n_params) + " parameters maximum with " + str(args.feature_maps) + " feature maps")

    if args.lr > 0:
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 0)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs // 3, 2 * args.epochs // 3], gamma=0.1)
    else:
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2 * args.epochs // 3], gamma=0.1)

    for epoch in range(args.epochs):
        train_data = train(model, epoch, args, device, train_loader, optimizer)
        test_data = test(model, args, device, test_loader)
        display_progress(epoch, train_data, test_data)
        torch.save(model, "/tmp/best_target_model.pt")
        scheduler.step()

    values = None
    for parameters in model.parameters():
        if values == None:
            if args.half:
                values = torch.abs(parameters.data.view(-1).half())
            else:
                values = torch.abs(parameters.data.view(-1))
        else:
            if args.half:
                values = torch.cat([values,torch.abs(parameters.data.view(-1).half())], dim=0)
            else:
                values = torch.cat([values,torch.abs(parameters.data.view(-1))], dim=0)
    values = torch.sort(values)[0]
    print("sorted {:d} values".format(values.shape[0]))
    
    perfs = []
    perfs_ft = []
    ths = []
    prunes = [int((1 - args.target) * 100000)]
    for i in prunes:
        print("Testing with pruning {:3d}/1000...                       ".format(i),end='')
        model = torch.load("/tmp/best_target_model.pt")
        th = values[((i * values.shape[0]) // 100000)]
        ths.append(th.item())
        print(str(th.item()) + " ", end='')
        for parameters in model.parameters():
            if args.half:
                parameters.data = parameters.data * (torch.abs(parameters.data) >= th).half()
            else:
                parameters.data = parameters.data * (torch.abs(parameters.data) >= th).float()
        masks = []
        res = test(model, args, device, test_loader)
        perfs.append(res["test_acc"])
        print(str(res["test_acc"]) + " ", end='')
        for parameters in model.parameters():
            if args.half:
                masks.append((torch.abs(parameters.data) >= th).half())
            else:
                masks.append((torch.abs(parameters.data) >= th).float())
        print("tuning")

        if args.lr > 0:
            optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 0)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.ft_epochs // 3, 2 * args.ft_epochs // 3], gamma=0.1)
        else:
            optimizer = optim.Adam(model.parameters())
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2 * args.ft_epochs // 3], gamma=0.1)

        for epoch in range(args.epochs, args.epochs + args.ft_epochs):
            train_data = train(model, epoch, args, device, train_loader, optimizer, masks = masks)
            test_data = test(model, args, device, test_loader)
            scheduler.step()
        res = test(model, args, device, test_loader)
        perfs_ft.append(res["test_acc"])
        print(" " + str(res["test_acc"]))
    
    values = {
        "\"dataset\": \"{:s}\"": args.dataset,
        "\"wd\": {:f}": args.wd,
        "\"a\": {:f}": args.a,
        "\"target\": {:f}": args.target,
        "\"epochs\": {:d}": args.epochs,
        "\"ft-epochs\": {:d}": args.ft_epochs,
        "\"feature_maps\": {:d}": args.feature_maps,
        "\"auto_augment\": {:b}": args.auto_augment,
        "\"cutout\": {:b}": args.cutout,
        "\"mixup\": {:b}": args.mixup,
        "\"cutmix\": {:b}": args.cutmix,
        "\"seed\": {:d}" : args.seed,
        "\"training_loss\": {:f}": train_data["train_loss"],
        "\"training_acc\": {:f}": train_data["train_acc"],
        "\"test_loss\": {:f}": test_data["test_loss"],
        "\"test_acc\": {:f}": test_data["test_acc"],
        "\"nparams\": {:d}": n_params,
        "\"ths\": {:s}": str(ths),
        "\"perfs\": {:s}": str(perfs),
        "\"perfs_ft\": {:s}": str(perfs_ft)
    }
    file_output = open("results_target.txt","a")
    file_output.write("results.append({")
    for key in values.keys():
        file_output.write(key.format(values[key]) + ", ")
    file_output.write("})\n")
    file_output.close()

if __name__ == '__main__':
        main()
