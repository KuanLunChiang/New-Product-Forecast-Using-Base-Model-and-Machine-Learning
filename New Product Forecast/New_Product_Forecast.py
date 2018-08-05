import pandas as pd
import numpy as np
import pyodbc
from GreyBass.Grey_Bass import Grey_Bass


con = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-964U6AM\MSSQLSERVER01;'
                      'Database=A1_supply;'
                      'Trusted_Connection=yes;')

recipe_ing = pd.read_sql("select * from vw_recipe_ingredient_qty", con)
salesNum = pd.read_sql("select * from vw_salesNum", con)
recing = pd.DataFrame(recipe_ing.pivot(index = 'RecipeId', columns = 'IngredientName', values = 'Quantity').fillna(0).to_records())

############### Meta Data ##########################################
data = salesNum.set_index('RecipeId').join(recing.set_index('RecipeId'))
data = data.sort_values(['date'])

############## Sales Data #########################################
salesData = data[['total_sales','date','MenuItemID']]
salesData = pd.DataFrame(salesData.groupby(['date','MenuItemID']).sum().to_records())


################ PQM Data ###############################################
from GreyBass.MachineLearning import BassInformation

ixList = salesData.MenuItemID.unique()

bi = BassInformation()
inputDict = {}
idList = {}
for i in ixList:
    if len(salesData.loc[salesData.MenuItemID == i].total_sales.tolist())>10:
        inputDict[i] = salesData.loc[salesData.MenuItemID == i].total_sales.tolist()
        idList[i] =i

bi.train_NLS(inputDict)
resFrame = pd.DataFrame({'MenuItemID':idList,'externalFactor':bi._externalFactor,'internalFactor':bi._internalFactor,'marketSize':bi._marketSize})

tempData = data.drop('total_sales',axis = 1).drop('date',axis = 1)
tempData = tempData.set_index('MenuItemID').drop_duplicates()
qpmFrame = resFrame.join(tempData,how = 'left').fillna(0)
qpmFrame.to_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
#######################################################################