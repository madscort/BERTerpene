#!/usr/bin/env python
import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt
import pandas as pd

import esm

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FASTA_PATH = "../data/allterp.faa" # Path to fasta
FASTA_CLASSES_PATH = "../data/terp.faa"
NEW_FASTA_PATH = "../data/neighbours_extended_terp.faa"
EMB_PATH = "../data/allterp_emb_esm2_33" # Path to directory of embeddings for fasta
EMB_CLASSES_PATH = "../data/terp_emb_esm2_33" # Path to directory of embeddings for fasta
EMB_LAYER = 33


# Our embeddings are stored with the file name from fasta header: {acc_num}_{category}.pt
ys = []
Xs = []
ys_tc = []
Xs_tc = []
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    category = header.split('_')[-1]
    ys.append(category)
    fn = f'{EMB_PATH}/{header}.pt'
    embs = torch.load(fn)
    Xs.append(embs['mean_representations'][EMB_LAYER])
    if 'tc' in category:
        ys_tc.append(header)
        fn = f'{EMB_PATH}/{header}.pt'
        embs = torch.load(fn)
        Xs_tc.append(embs['mean_representations'][EMB_LAYER])
Xs = torch.stack(Xs, dim=0).numpy()
Xs_tc = torch.stack(Xs_tc, dim=0).numpy()

# ### Reading only categorised sequences
ys_classes = []
Xs_classes = []

for header, _seq in esm.data.read_fasta(FASTA_CLASSES_PATH):
    category = header.split('_')[-1]
    ys_classes.append(category)
    fn = f'{EMB_CLASSES_PATH}/{header}.pt'
    embs = torch.load(fn)
    Xs_classes.append(embs['mean_representations'][EMB_LAYER])
Xs_classes = torch.stack(Xs_classes, dim=0).numpy()


train_size = 0.8
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, train_size=train_size, random_state=42)


# # ### Visualise Embeddings
categories_dict = dict.fromkeys(ys)
num = 0
for key in categories_dict:
    categories_dict[key] = num
    num += 1
    
categories_dict_no_tc = dict.fromkeys(ys_classes)
num = 1
for key in categories_dict_no_tc:
    categories_dict_no_tc[key] = num
    num += 1

ys_classes_numbers = []
for i in range(0, len(ys_classes)):
    ys_classes_numbers.append(categories_dict[ys_classes[i]])

# inv_categories = {v: k for k, v in categories_dict.items()}
inv_categories_no_tc = {v: k for k, v in categories_dict_no_tc.items()}

# ### Dataframes for one hot encoding
classes_for_one_hot = list(zip(ys_classes_numbers, ys_classes))
one_hot_array = OneHotEncoder().fit_transform(list(zip(list(categories_dict_no_tc.values()), list(categories_dict_no_tc.keys())))).toarray()
y = OneHotEncoder().fit_transform(classes_for_one_hot).toarray()


# ### Initialize grids for different regression techniques - doesn't work from here on
knn_grid = [
    {
        'model': [KNeighborsRegressor()],
        'model__n_neighbors': [5, 10],
        'model__weights': ['uniform', 'distance'],
        'model__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'model__leaf_size' : [15, 30],
        'model__p' : [1, 2],
    }
    ]

num_pca_components = 60
# make sure data preprocessing (PCA here) is run inside CV to avoid data leakage
pipe = Pipeline(
    steps = (
        ('pca', PCA(num_pca_components)),
        ('model', KNeighborsRegressor())
    )
)

result_list = []
grid_list = []
grid = GridSearchCV(
    estimator = pipe,
    param_grid = knn_grid,
    scoring = 'r2',
    verbose = 1,
    n_jobs = -1 # use all available cores

)
grid.fit(Xs_classes, y)
result_list.append(pd.DataFrame.from_dict(grid.cv_results_))
grid_list.append(grid)

print(grid)


# ### K Nearest Neighbors
result_list[0].sort_values('rank_test_score')[:5]

# ### Evaluation
for grid in grid_list:
    print(grid.best_estimator_.get_params()["steps"][1][1]) # get the model details from the estimator
    print()
    preds = grid.predict(Xs_test)
    # print(f'{scipy.stats.spearmanr(ys_test, preds)}')
    # print('\n', '-' * 80, '\n')
result_list[0].sort_values('rank_test_score')[:1].values

clustered = grid.predict(Xs_tc)

# ### Searching for matches between predictions and one_hot_encoding

matches = []

i = 0
for point in clustered.tolist()[1:]:
    if point in one_hot_array.tolist():
        matches.append([ys_tc[i], one_hot_array.tolist().index(point) + 1])
    i += 1
print('Number of new sequences: ', len(matches))


# ### Write all classified sequences to a new file
matches_dict = dict(matches)

outfile = open(NEW_FASTA_PATH, 'w')

for header, _seq in esm.data.read_fasta(FASTA_PATH):
    if 'tc' in header:
        if header in matches_dict:
            new_header = header + '_' + inv_categories_no_tc[matches_dict[header]]
            outfile.write(new_header + '\n')
            outfile.write(_seq)
            outfile.write('\n')
    else:
        outfile.write(header + '\n')
        outfile.write(_seq)
        outfile.write('\n')
        
        
outfile.close()
print('New file ready')

