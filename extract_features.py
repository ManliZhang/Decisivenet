from __future__ import print_function

import argparse
import csv
import os
import collections
import pickle
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import datasets

import torch.nn.functional as F

from os import path

use_gpu = torch.cuda.is_available()

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def extract_feature(novel_loader, model, checkpoint_name, is_pooled,l,c):
	save_dir = 'features/'
	tag_name = "pooled" if is_pooled else "binary"
	file_name = "{}_{}.plk".format(checkpoint_name,tag_name)
	save_file = save_dir + file_name
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	with torch.no_grad():
		output_dict = collections.defaultdict(list)

		for i, (inputs, labels) in enumerate(novel_loader):
            # compute output
			inputs = inputs.cuda()
			labels = labels.cuda()
			if is_pooled:
				outputs = model.temp_extract_pooled(inputs,l,c,0)
			else:
				outputs = model.temp_extract_features(inputs,l,c,0)
				
			outputs = outputs.cpu().data.numpy()
			result = torch.softmax(model.temp_forward(inputs,l,c,0),dim=1)
            
			for out, label in zip(outputs, labels):
				output_dict[label.item()].append(out)
    
		all_info = output_dict
		save_pickle(save_file, all_info)
		return all_info

if __name__ == '__main__':

    # Training settings
	parser = argparse.ArgumentParser(description='SWD')

    # Mandatory Arguments
	parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint_file')   
	parser.add_argument('--extract_pooled', action='store_true', default=False,
                        help='Extract pooled features instead of binary features')   
	
	args = parser.parse_args()
    
	base_file = "filelists/cifar/" + 'novel.json'
	checkpoint_name = args.checkpoint.split("/")[-1].replace(".pt","")
	l = int(args.checkpoint.split("_")[7])
	c = int(args.checkpoint.split("_")[1])
	
	print(args.checkpoint)
	print(checkpoint_name,l,c)
        
	_, novel_loader, _ = datasets.load_base_cifarfs(base_file=base_file, batch_size = 100,aug=False)
	model = torch.load(args.checkpoint)

	model = model.cuda()
	cudnn.benchmark = True
	model.eval()
	extract_feature(novel_loader,model,checkpoint_name,args.extract_pooled,l,c)