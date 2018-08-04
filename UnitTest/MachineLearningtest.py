import unittest
from GreyBass.MachineLearning import *
import numpy as np
import pandas as pd

class Test_MachineLearningtest(unittest.TestCase):

    def test_BassInformation_train_NLS(self):
        ## check input ##
        bi = BassInformation()
        self.assertRaises(ValueError,bi.train_NLS,"Invalid Input")
        self.assertRaises(ValueError,bi.train_NLS,[1])
        self.assertRaises(ValueError, bi.train_NLS,{'a':[1]})
        self.assertRaises(ValueError, bi.train_NLS, pd.DataFrame({'a':[1]}))

        ## Train Model ##
        xData = pd.DataFrame({'item1':np.arange(1,100,1),'item2':np.arange(2,101,1)})
        xData = {'item1':np.arange(1,100,1),'item2':np.arange(2,101,1)}
        bi.train_NLS(xData)
        print(bi._internalFactor, bi._externalFactor,bi._marketSize)

    def test_SVM_train(self):
        ## check input ##
        failxData = "Invalid Data"
        failyData = "Invalid Data"
        svm = SVM_Model()
        self.assertRaises(ValueError,svm.train,failxData,failyData)

        ## Train SVM ##
        xData = np.random.randn(500,10)
        yData = np.random.randn(500)
        svm.train(xData,yData)
        print(svm._model.support_vectors_)

    def test_SVM_predict(self):
        ## check input ##
        failxData = "Invalid Data"
        failyData = "Invalid Data"
        svm = SVM_Model()
        self.assertRaises(ValueError,svm.train,failxData,failyData)

        ## Train SVM ##
        xData = np.random.randn(500,10)
        yData = np.random.randn(500)
        svm.train(xData,yData)
        testData = np.random.randn(50,10)
        print(svm.predict(testData))
        self.assertRaises(ValueError,svm.predict,np.random.randn(1,1))

class Test_FeatureSelection (unittest.TestCase):
    def test_GA_fitness(self):
        ## Initialize ##
        svm = SVM_Model()
        ga = Genetic_Selection(svm)
        
        ## Check input ##
        self.assertRaises(ValueError,ga.fitness,'Invalid')
        self.assertRaises(ValueError,ga.fitness,1)
        


if __name__ == '__main__':
    unittest.main()
