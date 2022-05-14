import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import lightgbm as lgb 
from lightgbm import LGBMClassifier ##importing Light GBM classifier

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING ##hyperopt to perform hyperparameter optimization
from functools import partial

os.getcwd()
data = pd.read_csv("dataset_part_2.csv")
X = pd.read_csv('dataset_part_3.csv')
Y = data['Class'].to_numpy()
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])

space = {
            'n_estimators': hp.quniform('n_estimators', 500,1200,50),
            'num_leaves': hp.quniform('num_leaves', 3,50,1),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.19, 0.35, 0.01),
            'min_child_samples': hp.quniform('min_child_samples', 1, 100,1),
            'boosting_type':hp.choice('boosting_type',['gbdt', 'dart', 'goss']),
             'subsample': hp.quniform('subsample', 0.1, 0.95, 0.05),

            }
            
def objective(params):
    params = {
        'n_estimators':int(params[ 'n_estimators']),
        'num_leaves': int(params[ 'num_leaves']),
        'colsample_bytree': params[ 'colsample_bytree'],
        'min_child_samples':int(params[ 'min_child_samples']) ,
        'subsample':params['subsample'],
        'boosting_type': params['boosting_type']
    }
    
    clf = LGBMClassifier(objective  = 'binary',
                         class_weight ='balanced',
                         tree_method ='hist',
                         max_depth = -1,
                         n_jobs = -1,
                         min_split_gain=0.2,max_bin=250,
                         random_state = 27,**params) 
        
    clf.fit(X_train, Y_train)
    score_lgbm_best = clf.best_score_
    

    print("score_lgbm_best:", score_lgbm_best)

    return{'loss':1-score_lgbm_best, 'status': STATUS_OK }


 trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,trials=trials)   
            
print(best)

best_params_lgbm  = {'boosting_type': 'gbdt','colsample_bytree': 0.26, 'min_child_samples': 68, 'n_estimators': 500, 'num_leaves': 39,'subsample': 0.45}   

model_lgbm_best = LGBMClassifier(objective  = 'binary',
                         class_weight ='balanced',
                         tree_method ='hist',
                         max_depth = -1,
                         n_jobs = -1,
                         min_split_gain=0.2,max_bin=250,
                         random_state = 27,**best_params_lgbm)    

best_lgbm_gbdt_model_result = model_lgbm_best.fit(X_train, Y_train)  

yhat=best_lgbm_gbdt_model_result.predict(X_test)   
plot_confusion_matrix(Y_test,yhat)                    