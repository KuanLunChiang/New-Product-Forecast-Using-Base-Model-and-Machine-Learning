import pandas as pd
import numpy as np
import pyodbc
from GreyBass.Grey_Bass import Grey_Bass
from GreyBass.MachineLearning import *
import pickle

con = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-964U6AM\MSSQLSERVER01;'
                      'Database=A1_supply;'
                      'Trusted_Connection=yes;')

recipe_ing = pd.read_sql("select * from vw_recipe_ingredient_qty", con)
salesNum = pd.read_sql("select * from vw_salesNum", con)
salesNum['revenue'] = salesNum.Quantity * salesNum.Price
recing = pd.DataFrame(recipe_ing.pivot(index = 'RecipeId', columns = 'IngredientName', values = 'Quantity').fillna(0).to_records())

############### Meta Data ##########################################
data = salesNum.set_index('RecipeId').join(recing.set_index('RecipeId'))
data = data.sort_values(['date'])

############## Sales Data #########################################
salesData = data[['Quantity','date','MenuItemID','revenue']]
salesData = pd.DataFrame(salesData.groupby(['date','MenuItemID']).sum().to_records())
#salesData.groupby(['MenuItemID']).count().Quantity.plot()
#salesData.groupby(['date']).count().Quantity.plot()

############### Split Data ####################################################
"""
Training Data is set to be 2015-03-05 to 2015-05-15. Using two and a half months data to train model.
Test data is set to be 2015-05-16 to 2015-06-15. Using two months data for testing.
These data will further convert to competitive index for prediction.
"""

splitDate = pd.to_datetime("2015-05-15")
trainData = salesData.loc[salesData.date <=splitDate]
testData = salesData.loc[salesData.date > splitDate]


################ Train Data ###############################################
from GreyBass.MachineLearning import BassInformation


def pqm_df (xData):
    ixList = xData.MenuItemID.unique()

    bi = BassInformation()
    inputDict = {}
    idList = {}
    for i in ixList:
        if len(xData.loc[salesData.MenuItemID == i].Quantity.tolist())>10:
            inputDict[i] = xData.loc[xData.MenuItemID == i].Quantity.tolist()
            idList[i] =i

    bi.train_NLS(inputDict)
    resFrame = pd.DataFrame({'MenuItemID':idList,'externalFactor':bi._externalFactor,'internalFactor':bi._internalFactor,'marketSize':bi._marketSize})

    tempData = data.drop('Quantity',axis = 1).drop('date',axis = 1)
    tempData = tempData.set_index('MenuItemID').drop_duplicates()
    qpmFrame = resFrame.join(tempData,how = 'left').fillna(0) 
    return qpmFrame

trainData = pqm_df(trainData)

################### Train Bass Index External ###################################################
xData = trainData
xData = xData.loc[:,xData.max()>0]
xData = xData.loc[xData.sum(axis = 1) != 0,:]
yExtData = xData.externalFactor
yIntData = xData.internalFactor
yMarketData = xData.marketSize
xData = xData.drop(['revenue','marketSize','externalFactor','internalFactor','MenuItemID','Price'],axis = 1)

#################### Feature Selection #########################################################
"""
Filter quarter of the features using Bhattacharyya coefficient
and find the most divergent features
"""
selectFeatureNum = int(len(xData.columns)/2)
bs = Bhattacharyya_Selection(selectFeatureNum)
BFeature = bs.FeatureSelection(xData)
pd.DataFrame({'Bfeature':BFeature}).to_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\BFeature.csv")

"""
Normalize Data for all the distance model
Using Genetic Algorithm and different Machine Learning Models to find the
most valuable 10 features.
"""

xData = xData[BFeature]
xData = xData.loc[xData.sum(axis = 1) != 0,:]

svm = SVM_Model(regression=True,kernel = 'rbf',gamma = 'auto',C=1)
gp = Gaussian_Model(1e-10,n_jobs=-1)
rf = Forest_Model(n_estimators=20,n_jobs=-1)
knn = Adaptive_KNN_Model(radius= 10)

def GA_Selection (model,xData,yData,pop,iter):
    ga = Genetic_Selection(model,10,pop,0.3)
    GAFeatures = ga.FeatureSelection(xData,yData,'rmse',iter)
    return GAFeatures


'''
SVM
'''
print("Start SVM")
svm_ext_features = GA_Selection(svm,xData,yExtData,500,1000)
svm_int_features = GA_Selection(svm,xData,yIntData,500,1000)
svm_mrk_features = GA_Selection(svm,xData,yMarketData,500,1000)
print(svm_ext_features)
print(svm_int_features)
print(svm_mrk_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\svm_ext_features.npy",svm_ext_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\svm_int_features.npy",svm_int_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\svm_mrk_features.npy",svm_mrk_features_features)

'''
Adaptive KNN
'''
print("Start KNN")
knn_ext_features = GA_Selection(knn,xData,yExtData,500,1000)
knn_int_features = GA_Selection(knn,xData,yIntData,500,1000)
knn_mrk_features = GA_Selection(knn,xData,yMarketData,500,1000)
print(knn_ext_features)
print(knn_int_features)
print(knn_mrk_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\knn_ext_features.npy",knn_ext_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\knn_int_features.npy",knn_int_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\knn_mrk_features.npy",knn_mrk_features_features)


'''
Gaussian Process
'''
print("Start Gaussian")
gp_ext_features = GA_Selection(gp,xData,yExtData,200,1000)
gp_int_features = GA_Selection(gp,xData,yIntData,200,1000)
gp_mrk_features = GA_Selection(gp,xData,yMarketData,200,1000)
print(gp_ext_features)
print(gp_int_features)
print(gp_mrk_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\gp_ext_features.npy",gp_ext_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\gp_int_features.npy",gp_int_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\gp_mrk_features.npy",gp_mrk_features_features)

'''
Random Forest
'''
print("Start Random Forest")
rf_ext_features = GA_Selection(rf,xData,yExtData,200,1000)
rf_int_features = GA_Selection(rf,xData,yIntData,200,1000)
rf_mrk_features = GA_Selection(rf,xData,yMarketData,200,1000)
print(rf_ext_features)
print(rf_int_features)
print(rf_mrk_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\rf_ext_features.npy",rf_ext_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\rf_int_features.npy",rf_int_features)
np.save(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\rf_mrk_features.npy",rf_mrk_features)



####################### Train Bass Index Model ################################################
"""
Here will train all the model and select the best two models for ensemble learning.
The output will be three ensembled predictoin model for each single Bass Index Coefficients.
"""
from sklearn.model_selection import KFold

def kFoldCV (xData,yData,selectList):
    train = xData[selectList['selection']]
    kf = KFold(5,True)
    x_train = []
    x_test = [] 
    y_train = []
    y_test = []
    for train_index, test_index in kf.split(train):
        x_train.append(train.iloc[train_index])
        x_test.append(train.iloc[test_index])
        y_train.append(yData.iloc[train_index])
        y_test.append(yData.iloc[test_index])
    return x_train,y_train,x_test,y_test


def cvScore (mdl,x_train,y_train,x_test,y_test):
    score = 0
    for i in range(5):
        mdl.train(x_train[i],y_train[i])
        prd = mdl.predict(x_test[i])
        real = np.array(y_test[i].tolist())
        score += mdl.score(prd,real)
    score = score/5
    return score


def gridSearch(xData,yData,selectList,mdl,paraList,para):
    x_train,y_train,x_test,y_test = kFoldCV(xData,yData,selectList)
    gridList = {'paraName':para,'para':None,'cvScore':0}
    score = -9999999999
    res = None
    for i in paraList:
        mdl._model.set_params(**{para:i})
        temp = cvScore(mdl,x_train,y_train,x_test,y_test)
        res = mdl

        if temp < score or score == -9999999999:
            score = temp
            gridList['para'] = i
            gridList['cvScore'] = temp
            res = mdl
    return gridList,res

"""
NLS Model
"""

'''
SVM
'''
svm = SVM_Model()
svm_ext_c,svm_ext_tune = gridSearch(xData,yExtData,svm_ext_features,svm,np.arange(0.000001,0.0001,0.000001),'C')
svm_ext_g,svm_ext_tune = gridSearch(xData,yExtData,svm_ext_features,svm,np.arange(0.00001,0.0001,0.00001),'gamma')

svm_int_c,svm_int_tune = gridSearch(xData,yIntData,svm_int_features,svm,np.arange(0.00001,1,0.00001),'C')
svm_int_g,svm_int_tune = gridSearch(xData,yIntData,svm_int_features,svm,np.arange(0.00001,1,0.00001),'gamma')

svm_mrk_c,svm_mrk_tune = gridSearch(xData,yMarketData,svm_mrk_features,svm,np.arange(0.0001,1,0.0001),'C')
svm_mrk_g,svm_mrk_tune = gridSearch(xData,yMarketData,svm_mrk_features,svm,np.arange(0.00001,0.0001,0.00001),'gamma')

print(svm_ext_c,'\n',svm_ext_g)
print(svm_int_c,'\n',svm_int_g)
print(svm_mrk_c,'\n',svm_mrk_g)

'''
Adaptive KNN
'''

knn = Adaptive_KNN_Model(True)
knn_ext_r,knn_ext_tune = gridSearch(xData,yExtData,knn_ext_features,knn,np.arange(10,100,1),'radius')
knn_int_r,knn_int_tune = gridSearch(xData,yIntData,knn_int_features,knn,np.arange(10,100,1),'radius')
knn_mrk_r,knn_mrk_tune = gridSearch(xData,yMarketData,knn_mrk_features,knn,np.arange(10,100,1),'radius')

print(knn_ext_r)
print(knn_int_r)
print(knn_mrk_r)

'''
Gaussian Process
'''




####################### Competitive Analysis Model ################################################################
"""
This section will train the competitive index model 
using NLS coefficients and ingredients
"""
