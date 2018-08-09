import numpy as np
import scipy as sci
import pandas as pd
from scipy.optimize import least_squares
from GreyBass import Grey_Bass

gb = Grey_Bass.Grey_Bass()
testData = np.array([1,2,3,4,5])
res = gb._NLS(testData)

########################################
import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)