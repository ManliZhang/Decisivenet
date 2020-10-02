import torch
from auto_augment import Cutout, AutoAugment
from torchvision import datasets, transforms
from abc import ABCMeta, abstractmethod
identity = lambda x:x
import json
import os
from PIL import Image

"""
"""
def load_mnist(args, **kwargs):
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs
    )

    metadata = {
        "input_shape" : (1,28,28),
        "n_classes" : 10
    }

    return train_loader, test_loader, metadata

"""
"""
def load_fashionMnist(args, **kwargs):
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data/fashionMnist', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('data/fashionMnist', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    metadata = {
        "input_shape" : (1,28,28),
        "n_classes" : 10
    }

    return train_loader, test_loader, metadata

"""
"""
def load_cifar10(args, **kwargs):
    list_trans = [        
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
        ]

    if args.auto_augment:
        list_trans.append(AutoAugment())
    if args.cutout:
        list_trans.append(Cutout())

    list_trans.append(transforms.ToTensor())
    list_trans.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    transform_train = transforms.Compose(list_trans)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs, num_workers = 4)

    validation_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs, num_workers = 4)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs, num_workers = 4)

    metadata = {
        "input_shape" : (3,32,32),
        "n_classes" : 10
    }

    return train_loader, test_loader, metadata

"""
"""
def load_cifar100(args, **kwargs):
    list_trans = [        
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
        ]

    if args.auto_augment:
        list_trans.append(AutoAugment())
    if args.cutout:
        list_trans.append(Cutout())

    list_trans.append(transforms.ToTensor())
    list_trans.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    transform_train = transforms.Compose(list_trans)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs, num_workers = 4)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, download=True, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs, num_workers = 4)

    metadata = {
        "input_shape" : (3,32,32),
        "n_classes" : 100
    }

    return train_loader, test_loader, metadata

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])):
        self.image_size = image_size
        self.normalize_param = normalize_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

    
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

def load_base_cifarfs(base_file,batch_size,aug=True):
    base_datamgr    = SimpleDataManager(32, batch_size = batch_size)
    base_loader     = base_datamgr.get_data_loader( base_file , aug=aug )
    base_datamgr_test    = SimpleDataManager(32, batch_size = batch_size)
    base_loader_test     = base_datamgr_test.get_data_loader( base_file , aug = False )
    metadata = {
        "input_shape" : (3,32,32),
        "n_classes" : 64
    }
    return base_loader, base_loader_test, metadata


"""
"""
def get_data_loaders(args, **kwargs):

    dataset_name = args.dataset.lower()
    base_file = "filelists/cifar/base.json"
    val_file = "filelists/cifar/val.json"

    if dataset_name=="mnist":
        return load_mnist(args, **kwargs)

    elif dataset_name=="fashionmnist":
        return load_fashionMnist(args, **kwargs)

    elif dataset_name=="cifar10":
        return load_cifar10(args, **kwargs)

    elif dataset_name=="cifar100":
        return load_cifar100(args, **kwargs)
    
    elif dataset_name =="cifarfs-base":
        return load_base_cifarfs(base_file=base_file, batch_size = args.batch_size,**kwargs)

    else :
        raise Exception("Dataset '{}' is no recognized dataset. Could not load any data.".format(args.dataset))
