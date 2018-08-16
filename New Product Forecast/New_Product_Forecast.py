import pandas as pd
import numpy as np
import pyodbc
from GreyBass.Grey_Bass import Grey_Bass
from GreyBass.MachineLearning import *

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
yExtData = trainData.externalFactor
xExtData = trainData.drop(['revenue','marketSize','externalFactor','internalFactor','MenuItemID','Price'],axis = 1)

#################### Feature Selection #########################################################
"""
Filter quarter of the features using Bhattacharyya coefficient
and find the most divergent features
"""
selectFeatureNum = int(len(xExtData.columns)/4)
bs = Bhattacharyya_Selection(selectFeatureNum)
BFeature = bs.FeatureSelection(xExtData)
pd.DataFrame({'Bfeature':BFeature}).to_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\BFeature.csv")

"""
Using Genetic Algorithm and different Machine Learning Models to find the
most valuable 10 features.
"""

svm = SVM_Model(regression=True,kernel = 'rbf',gamma = 'auto',C=1)
gp = Gaussian_Model(1e-10)

def GA_Selection (model,xData,yData,pop,iter):
    ga = Genetic_Selection(model,10,pop,0.3)
    GAFeatures = ga.FeatureSelection(xData,yData,'rmse',iter)
    return GAFeatures

svm_ext_features = GA_Selection(svm,xExtData[BFeature],yExtData,100,1000)


####################### Train Bass Index Model ################################################
"""
Here will train all the model and select the best two models for ensemble learning.
The output will be three ensembled predictoin model for each single Bass Index Coefficients.
"""
from sklearn.model_selection import KFold
train = xExtData[svm_ext_features['selection']]
kf = KFold(5,True)
x_train = []
x_test = [] 
y_train = []
y_test = []
for train_index, test_index in kf.split(train):
    x_train.append(train.iloc[train_index])
    x_test.append(train.iloc[test_index])
    y_train.append(yExtData.iloc[train_index])
    y_test.append(yExtData.iloc[test_index])


svm = SVM_Model(regression=True,kernel = 'rbf',gamma = 'auto',C=1)
score = 0
for i in range(5):
    svm.train(x_train[i],y_train[i])
    prd = svm.predict(x_test[i])
    real = np.array(y_test[i].tolist())
    score += svm.score(prd,real)
score = score/5




####################### Competitive Analysis Model ################################################################
"""
This section will train the competitive index model 
using NLS coefficients and ingredients
"""