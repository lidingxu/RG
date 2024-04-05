# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:08:28 2024

@author: crist
"""
# TO BE CHANGED
pathfolder = 'C:/Users/crist/Desktop/PARIS/RuleSets/TEST/RG/'

approach = 'Convex'
#approach = 'Nonconvex'

print('-------------------')
print('Approach:', approach)

import warnings
warnings.filterwarnings("ignore")


from sklearn.metrics import accuracy_score

import os
from features import FeatureBinarizer
from boolean_rule_cg_nonconvex import BooleanRuleCGNonconvex
from boolean_rule_cg_convex import BooleanRuleCGConvex
from BRCG import  BRCGExplainer

import numpy as np
import pandas as pd
import timeit
import shelve
import random


pathcode = pathfolder + 'src/'
pathresults = pathfolder + 'results/'
os.chdir(pathcode)

from ucimlrepo import fetch_ucirepo
# from ucimlrepo import list_available_datasets
# check which datasets can be imported
# list_available_datasets()

for dataname in ['Banknote Authentication',
                 'Ionosphere',
                 'Breast Cancer Wisconsin (Diagnostic)',
                 'Blood Transfusion Service Center',
                 'Tic-Tac-Toe Endgame',
                 'Adult',
                 'Bank Marketing',
                 'MAGIC Gamma Telescope',
                 'Mushroom',
                 'Musk (Version 2)']:
    print('-------------------')
    print(dataname)
    print('-------------------')
    # Read data
    dataset = fetch_ucirepo(name = dataname)  
    X = dataset.data.features
    y = np.array((dataset.data.targets == np.unique(dataset.data.targets)[0])*1)
    y = np.reshape(y,len(y))
    column_names = dataset.data.features.columns
    
    X_df = pd.DataFrame(X, columns=column_names)
    X_df = X_df.dropna()

    from sklearn.model_selection import KFold
    k = 5
    cv = KFold(n_splits=k, shuffle=True, random_state=1)
    ### SET PARAMETERS ###
    lambda0 = np.linspace(0.001, 0.01, 5)
    lambda1 = np.linspace(0.001, 0.01, 5)
    ######################
    n_lambda0 = len(lambda0)
    n_lambda1 = len(lambda1)
    n = n_lambda0*n_lambda1
    acc_train = np.zeros((k,n))
    acc_test = np.zeros((k,n))
    nrules = np.zeros((k,n))
    nconditions = np.zeros((k,n))
    time = np.zeros((k,n))
    models = [[None]*(n)]*k
    fold = 0
    for train_index, test_index in cv.split(X_df):
        print('fold = ', fold+1)
        print('-------------------')
        X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fb = FeatureBinarizer(negations=True)
        X_train_fb = fb.fit_transform(X_train)
        X_test_fb = fb.transform(X_test)
        ind = 0
        for i in range(n_lambda0):
            i = 0
            lambda0n = lambda0[i]
            for j in range(n_lambda1):
                j = 0
                lambda1n = lambda1[j]
                print('lambda_0 =', lambda0n)
                print('lambda_1 =', lambda1n)
                random.seed(1)
                np.random.seed(1)                
                t0=timeit.default_timer()
                
                if approach == 'Convex':
                    boolean_model = BooleanRuleCGConvex(silent = True, lambda0=lambda0n, lambda1=lambda1n)
                else:
                    boolean_model = BooleanRuleCGNonconvex(silent = True, lambda0=lambda0n, lambda1=lambda1n)
                explainer = BRCGExplainer(boolean_model)
                explainer.fit(X_train_fb, y_train)

                t1=timeit.default_timer()
                time[fold,ind] = t1-t0
                print('\tTook %0.3fs to complete the whole process' % (time[fold,ind]))
                
                explanation = explainer.explain()
                nrules[fold,ind] = int(len(explanation['rules']))
                nconditions[fold,ind] = int(np.sum([explanation['rules'][j].count('AND')+1 for j in range(int(nrules[fold,ind]))]))
                for exp in range(int(nrules[fold,ind])):
                    print('rule ', int(exp+1),': ', explanation['rules'][exp])
                
                Y_pred_train = explainer.predict(X_train_fb)
                acc_train[fold,ind] = accuracy_score(y_train, Y_pred_train)
                print('acc_train', acc_train[fold,ind])
                
                Y_pred = explainer.predict(X_test_fb)
                acc_test[fold,ind] = accuracy_score(y_test, Y_pred)
                print('acc_test', acc_test[fold,ind])
                
                print('-------------------')
        
                models[fold][ind] = explanation
                ind = ind + 1
        fold = fold + 1
    n_samples = int(len(X_train_fb) + len(X_test_fb))
    print('n_samples =', n_samples)
    n_features = int(len(X_train_fb.iloc[0]))
    print('n_features =', n_features)
    accuracy_train_mean = np.mean(acc_train,axis=0)
    print('accuracy_train_mean', accuracy_train_mean)
    accuracy_test_mean = np.mean(acc_test,axis=0)
    print('accuracy_test_mean', accuracy_test_mean)
    nrules_mean = np.mean(nrules,axis=0)
    print('nrules_mean', nrules_mean)
    nconditions_mean = np.mean(nconditions,axis=0)
    print('nconditions_mean', nconditions_mean)
    time_mean = np.mean(time,axis=0)
    print('time_mean', time_mean)
    
    shelf = shelve.open(pathresults + "results_" + approach + '_' + dataname + ".dat")
    shelf["accuracy_train_mean"] = accuracy_train_mean
    shelf["accuracy_test_mean"] = accuracy_test_mean
    shelf["nrules_mean"] = nrules_mean
    shelf["nconditions_mean"] = nconditions_mean
    shelf["time_mean"] = time_mean
    shelf["acc_train"] = acc_train
    shelf["acc_test"] = acc_test
    shelf["nrules"] = nrules
    shelf["nconditions"] = nconditions
    shelf["time"] = time
    shelf["lambda0"] = lambda0
    shelf["lambda1"] = lambda1
    shelf["dataname"] = dataname
    shelf["models"] = models
    shelf.close()