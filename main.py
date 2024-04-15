# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:08:28 2024

@author: Liding Xu
"""

import os
import numpy as np
import pandas as pd
import timeit
import json
import random
from scipy.stats.mstats import gmean
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
# from ucimlrepo import list_available_datasets

from utils import *
from RG import *


# check which datasets can be imported
# list_available_datasets()

def main():
    global args
    
    save_path = os.path.join(args.results_dir, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Read data
    dataset = fetch_ucirepo(name = args.dataset)  
    X = dataset.data.features
    y = np.array((dataset.data.targets == np.unique(dataset.data.targets)[0])*1)
    y = np.reshape(y,len(y))
    column_names = dataset.data.features.columns

    # Partition data
    X_df = pd.DataFrame(X, columns=column_names)
    X_df = X_df.dropna()
    k = 5
    cv = KFold(n_splits=k, shuffle=True, random_state=args.seed)

    np.random.seed(args.seed)
    # Result holders
    
    print(args) 
    results = [FoldResult() for fold in range(k)]
    for fold, (train_index, test_index) in enumerate(cv.split(X_df)):
        print('fold = ', fold+1)
        print('-------------------')
        X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fb = FeatureBinarizer(negations=True)
        X_train_fb = fb.fit_transform(X_train)
        X_test_fb = fb.transform(X_test)
               
        t0=timeit.default_timer()

        if args.model == 'convex':
            boolean_model = BooleanRuleCGConvex(lambda0 = args.lambda0, lambda1 = args.lambda1, silent = args.verbose == 0)
        elif args.model == 'nonconvex':
            boolean_model = BooleanRuleCGNonconvex(lambda0 = args.lambda0, lambda1 = args.lambda1, silent = args.verbose == 0)
        elif args.model == 'dc':
            boolean_model = BooleanRuleCGDC(lambda0 = args.lambda0, lambda1 = args.lambda1, silent = args.verbose == 0)
        explainer = BRCGExplainer(boolean_model)
        explainer.fit(X_train_fb, y_train)

        t1=timeit.default_timer()

        results[fold].time = t1-t0
        print('\tTook %0.3fs to complete the whole process' % (results[fold].time))
        
        results[fold].loss = explainer.statistics()
        explanation = explainer.explain()
        results[fold].nrules = int(len(explanation['rules']))
        results[fold].nconditions= int(np.sum([explanation['rules'][j].count('AND')+1 for j in range(results[fold].nrules)]))
        for exp in range(int(results[fold].nrules)):
            print('rule ', int(exp+1),': ', explanation['rules'][exp])
        
        Y_pred_train = explainer.predict(X_train_fb)
        results[fold].acc_train = accuracy_score(y_train, Y_pred_train)
        print('acc_train', results[fold].acc_train)
        
        Y_pred = explainer.predict(X_test_fb)
        results[fold].acc_test = accuracy_score(y_test, Y_pred)
        print('acc_test', results[fold].acc_test)
        
        print('-------------------')
    
    result = TestRunResult()
    result.model = args.model
    result.dataset = args.dataset
    result.lambda0 = args.lambda0
    result.lambda1 = args.lambda1
    result.n_samples = int(len(X_train_fb) + len(X_test_fb))
    print('n_samples =', result.n_samples)
    result.n_features = int(len(X_train_fb.iloc[0]))
    print('n_features =', result.n_features)

    dict_result = {}
    avg_attrs = ["loss", "acc_train", "acc_test", "nrules", "nconditions", "time"]
    for attr in avg_attrs:
        lst = [getattr(foldresult, attr) for foldresult in results]
        setattr(result, attr +"_mean", gmean(lst)) 

    dict_result = {}
    for attr in dir(result):
        if not (attr.startswith('__') and attr.endswith('__')) and not callable(getattr(result, attr)):
            dict_result[attr] =  getattr(result, attr) 
    json_str = json.dumps(dict_result, indent=4)
    save_file=os.path.join(save_path, str(args.model) + "_" + str(args.lambda0) +  "_" + str(args.lambda1) + ".json")
    print(save_file)
    with open(save_file, "w") as json_file:
        json_file.write(json_str)

if __name__ == '__main__':
    main()
