import os
import pandas as pd
import test_features
import argparse
import tqdm

parser = argparse.ArgumentParser(
        description='Benchmark features')
parser.add_argument('--runs', type=int, default=1, help='Number of few-shot iterations')
parser.add_argument('--hadamard', action='store_true', default=False,
                            help='Use hadamard distance for 1NN')   
parser.add_argument('--not_normalize', action='store_true', default=False,
                            help='Dont normalize l2 distance')   

args = parser.parse_args()

path_features = "features/"
list_dicts = list()
files = list()
for filename in os.listdir(path_features):
    path_file = path_features+filename
    treat_file = filename.replace(".plk","")
    network = treat_file.split("_")[9]
    if "resnet50" in network:
        files.append(path_file)

for path_file in tqdm.tqdm(files):
    filename = path_file.split("/")[1]
    treat_file = filename.replace(".plk","")
    network = treat_file.split("_")[9]        
    l = int(treat_file.split("_")[7])
    _type = treat_file.split("_")[-1]
    mean_1nn, confiance_1nn, mean_ncm, confiance_ncm = test_features.test_data(path_file,args.runs,args.hadamard,not args.not_normalize, False)
    file_dict = dict(l=l,features=_type,mean_1nn=mean_1nn, confiance_1nn=confiance_1nn, mean_ncm=mean_ncm, confiance_ncm=confiance_ncm)
    list_dicts.append(file_dict)
df = pd.DataFrame(list_dicts).sort_values(["l","features"])
print(df.to_string(index=False))
df.to_pickle("benchmark_few_shot_50.pkl")
