import torch
from auto_augment import Cutout, AutoAugment
from torchvision import datasets, transforms

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


"""
"""
def get_data_loaders(args, **kwargs):

    dataset_name = args.dataset.lower()

    if dataset_name=="mnist":
        return load_mnist(args, **kwargs)

    elif dataset_name=="fashionmnist":
        return load_fashionMnist(args, **kwargs)

    elif dataset_name=="cifar10":
        return load_cifar10(args, **kwargs)

    elif dataset_name=="cifar100":
        return load_cifar100(args, **kwargs)

    else :
        raise Exception("Dataset '{}' is no recognized dataset. Could not load any data.".format(args.dataset))
