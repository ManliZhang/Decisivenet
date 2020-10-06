import argparse
import pickle
import torch
import tqdm
import random
import numpy as np

#from https://github.com/yhu01/transfer-sgc
def sample_case(ld_dict, shot, n_ways, n_queries):
    sample_class = random.sample(list(ld_dict.keys()), n_ways)
    train_input = []
    test_input = []
    for each_class in sample_class:
        samples = random.sample(ld_dict[each_class], shot + n_queries)
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input

#from https://github.com/yhu01/transfer-sgc
def get_labels(num_ways, shot, num_queries):
    train_labels = []
    test_labels = []
    classes = [i for i in range(num_ways)]
    for each_class in classes:
        train_labels += [each_class] * shot
        test_labels += [each_class] * num_queries

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_labels, test_labels

#from https://github.com/yhu01/transfer-sgc
def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return np.round(m, 2), np.round(pm, 2)


def nearest_mean_classifier(train_set, train_labels, test_set, test_labels,normalize=False):
    # Compute the means of the feature vectors of the same classes
    n_way = torch.max(train_labels) + 1
    means = torch.zeros(n_way, train_set.shape[1]).cuda()
    for label in range(n_way):
        means[label] = torch.mean(train_set[train_labels == label], dim=0)
    if normalize:
        means = torch.nn.functional.normalize(means, dim=1, p=2)
    distances = torch.cdist(test_set, means)
#    print(distances.shape)
    # Choose the labels according to the closest mean
    predicted = torch.argmin(distances, dim=1)
    # Compute accuracy
    total = test_labels.shape[0]
    correct = (predicted == test_labels).sum()
    test_acc = (100. * correct / total).item()
    return np.round(test_acc, 2)


def nearest_neighbor_classifier(x_train, x_test, y_train, y_test,hadamard=False):
    p = 2 if not hadamard else 0
    distances = torch.cdist(x_test, x_train,p=p).cpu().numpy()
#    print(distances.shape)
    ranks = np.argmin(distances, axis=1)
    accuracy = ((y_train[ranks] == y_test).sum())/y_test.shape[0]
    result = accuracy
    result = np.round(100*result, 2)
    return result

def test_data(path_file,runs,hadamard=False,normalize=True,verbose=False):
    examples_per_class = 5
    num_classes = 5
    num_queries = 595

    with open(path_file, 'rb') as f:
        feature_dict = pickle.load(f)

    results_1nn, results_ncm = list(), list()
    random.seed(0) #Fix seed for the random generator - Reproducibility
    loop = tqdm.tqdm(range(runs)) if verbose else range(runs)
    for run in loop:
        train_data, test_data = sample_case(feature_dict, examples_per_class, num_classes, num_queries)
        y_train, y_test = get_labels(num_classes, examples_per_class, num_queries)
        x_train = torch.cuda.FloatTensor(train_data)
        x_test = torch.cuda.FloatTensor(test_data)

        if normalize:
            x_train = torch.nn.functional.normalize(x_train,dim=1,p=2)
            x_test = torch.nn.functional.normalize(x_test,dim=1,p=2)

        results_1nn.append(nearest_neighbor_classifier(
            x_train, x_test, y_train, y_test,hadamard=hadamard))
        results_ncm.append(nearest_mean_classifier(x_train, torch.cuda.LongTensor(
            y_train), x_test, torch.cuda.LongTensor(y_test),normalize=normalize))
    #    if run == 0:
    #        print("Shape train {}, shape test {}".format(x_train.shape,x_test.shape))

    mean_1nn, confiance_1nn = compute_confidence_interval(results_1nn)
    mean_ncm, confiance_ncm = compute_confidence_interval(results_ncm)

    if verbose:
        print("Results 1nn {}+-{}, NCM {}+-{}".format(mean_1nn, confiance_1nn, mean_ncm, confiance_ncm))
    return mean_1nn, confiance_1nn, mean_ncm, confiance_ncm
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Denoising DNN features with class low-pass graph filters test on the few-shot task')
    parser.add_argument('--runs', type=int, default=100, help='Number of few-shot iterations')
    parser.add_argument('--data', type=str, required=True,
                            help='data file')   
    parser.add_argument('--hadamard', action='store_true', default=False,
                            help='Use hadamard distance for 1NN')   
    parser.add_argument('--normalize', action='store_true', default=False,
                            help='Use normalized l2 distance')   


    args = parser.parse_args()
    test_data(args.data,args.runs,args.hadamard,args.normalize,True)
    



