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

############### Property Data #################################################
pd1 = salesData.loc[salesData.MenuItemID == 2]
pd1.total_sales.cumsum()

################ Grey Bass ###############################################
gb = Grey_Bass()
pd1 = salesData.loc[salesData.MenuItemID == 2]
input = pd1.total_sales.tolist()[0:20]
gb._NLS(input)

