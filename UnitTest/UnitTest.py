import numpy as np
import scipy as sci
import pandas as pd
from scipy.optimize import least_squares
from GreyBass import Grey_Bass

gb = Grey_Bass.Grey_Bass()
testData = np.array([1,2,3,4,5])
res = gb._NLS(testData)