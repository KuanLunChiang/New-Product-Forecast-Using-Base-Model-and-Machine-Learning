import unittest
from GreyBass.MachineLearning import *
import numpy as np
import pandas as pd


class Test_BassInfo(unittest.TestCase):
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



class Test_SVM(unittest.TestCase):

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

    def test_SVM_score(self):
        self.fail()




class Test_GeneticAlgorithm(unittest.TestCase):

    def test_GA_fitness(self):
        ## Initialize ##
        svm = SVM_Model()
        ga = Genetic_Selection(svm,20,100)
        
        ## Check input ##
        self.assertRaises(ValueError,ga.fitness,[1,1],[1,1],'Invalid')
        self.assertRaises(ValueError,ga.fitness,[1,1],[1,1],1)
        ## Fitness ##
        fit = ga.fitness([1,1],[2,1],'rmse')
        ans = 1 / np.sqrt(1 / 2)
        self.assertEqual(ans,fit)
        fit = ga.fitness([1,1],[2,1],'classificationError')
        self.assertEqual(1 - 1 / 2,fit)

    def test_GA_generatePop(self):
        ## Initialize ##
        svm = SVM_Model()
        ga = Genetic_Selection(svm,30,500)
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        xData = df.drop(['MenuItemID','externalFactor','internalFactor','marketSize'],axis =1)
        featureNum = len(xData.columns)
        ## Check input ##
        self.assertRaises(ValueError,ga._generatePopulation,'Invalid')
        res = ga._generatePopulation(featureNum)
        self.assertEqual(len(res),500)
        self.assertEqual(len(res[1]),featureNum)
        print(res)
        for i in res:
            self.assertEqual(sum(i),30)
    def test_GA_selection(self):
        ## Initialize ##
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        xData = df.drop(['MenuItemID','externalFactor','internalFactor','marketSize'],axis =1)
        yData = df.externalFactor
        svm = SVM_Model()
        ga = Genetic_Selection(svm,20,1000)

        ## check input ##
        self.assertRaises(ValueError,ga.selection,xData,yData,'Invalid',1,'invalid')
        self.assertRaises(ValueError,ga.selection,xData,yData,[100],"Invalid",'rmse')
        ## selection ##
        featureNum = len(xData.columns)
        pop = ga._generatePopulation(featureNum)
        sel = ga.selection(xData,yData,pop,200,'rmse')
        print(sel)
        print(ga._elite)
        self.assertEqual(len(sel),200)

    def test_GA_crossover(self):
        ## Initialize ##
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        xData = df.drop(['MenuItemID','externalFactor','internalFactor','marketSize'],axis =1)
        yData = df.externalFactor
        svm = SVM_Model()
        ga = Genetic_Selection(svm,20,100)
        featureNum = len(xData.columns)
        pop = ga._generatePopulation(featureNum)
        sel = ga.selection(xData,yData,pop,20,'rmse')
        
        ## check input ##
        self.assertRaises(ValueError,ga.crossover,"invalid value")
        self.assertRaises(ValueError,ga.crossover,{'a':123,'b':123})

        ## crossover ##
        nextG = ga.crossover(sel)
        for i in nextG:
            self.assertEqual(sum(i),20)
            self.assertEqual(len(i),featureNum)
        self.assertEqual(len(nextG),(len(sel)-1)*2)
        print(nextG)

    def test_GA_breed(self):
        svm = SVM_Model()
        ga = Genetic_Selection(svm,20,100)
        pa = np.array([0,0,0,1,1,1,0])
        pb = np.array([1,0,0,0,0,1,1])
        childa ,childb = ga._breed(pa,pb)
        self.assertEqual(sum(childa),3)
        self.assertEqual(sum(childb),3)
        self.assertEqual(len(childa),len(childb))
        pa = np.array([0,0,0,1,1,1,0,0])
        pb = np.array([1,0,0,0,0,1,1,0])
        childa ,childb = ga._breed(pa,pb)
        self.assertEqual(sum(childa),3)
        self.assertEqual(sum(childb),3)
        self.assertEqual(len(childa),len(childb))
        print(childa,childb)

    def test_GA_mutate (self):
        ## Initialize ##
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        xData = df.drop(['MenuItemID','externalFactor','internalFactor','marketSize'],axis =1)
        yData = df.externalFactor
        svm = SVM_Model()
        ga = Genetic_Selection(svm,20,100,0.3)
        featureNum = len(xData.columns)
        pop = ga._generatePopulation(featureNum)
        sel = ga.selection(xData,yData,pop,100,'rmse')
        nextG = ga.crossover(sel)
        ## check input ##
        self.assertRaises(ValueError,ga.mutation,"Invalid Value")
        self.assertRaises(ValueError,ga.mutation,5)
        self.assertRaises(ValueError,ga.mutation,{'ab':2})
        ## mutate ##
        nextGeneration = ga.mutation(nextG)
        for i in nextGeneration:
            self.assertEqual(sum(i),20)
            self.assertEqual(len(i),featureNum)

    def test_GA_randomSwitch(self):
        svm = SVM_Model()
        ga = Genetic_Selection(svm,20,100)
        pa = np.array([0,0,0,1,1,1,0])
        sw = ga._randomSwitch(pa)
        self.assertEqual(sum(sw),sum(pa))
        self.assertEqual(len(sw),len(pa))
        for i in range(len(sw)):
            if sw[i]==1:
                self.assertNotEqual(sw[i],pa[i])

    def test_GA_FeatureSelection (self):
        ## Initialize ##
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        xData = df.drop(['MenuItemID','externalFactor','internalFactor','marketSize'],axis =1)
        yData = df.marketSize
        svm = SVM_Model(kernel='rbf',gamma='auto',C =0.1)
        ga = Genetic_Selection(svm,15,500,0.5)
        features = ga.FeatureSelection(xData,yData,'rmse',300)
        print(features)


class Test_Bhattacharyya_Selection (unittest.TestCase):
    
    def test_BC_BhattacharyyaCoef (self):
        ## Initialize ##
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        xData = df
        bs = Bhattacharyya_Selection(None)
        
        ## Check input ##
        self.assertRaises(ValueError,bs.Bhattacharyya_coefficient,"Invalid Input","Not valid")
        self.assertRaises(ValueError,bs.Bhattacharyya_coefficient,{'a':1},{'b':2})
        self.assertRaises(ValueError,bs.Bhattacharyya_coefficient,2,2)

        ## Coefficient ##
        coef = bs.Bhattacharyya_coefficient(xData.internalFactor,xData.externalFactor)
        print(coef)



    def test_BC_standardDist (self):       
        ## Initialize ##
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        #xData = df.drop(['MenuItemID','externalFactor','internalFactor','marketSize'],axis =1)
        xData = df
        bs = Bhattacharyya_Selection(None)
        
        ## Check input ##
        self.assertRaises(ValueError,bs._standardize,"Invalid Input")
        self.assertRaises(ValueError,bs._standardize,{'a':1})
        self.assertRaises(ValueError,bs._standardize,2)
        
        ## Scale ##
        res = bs._standardize(xData.externalFactor)
        self.assertEqual(len(res),len(xData.externalFactor))
        self.assertAlmostEqual(max(res),1)
        for i in res:
            self.assertGreaterEqual(i,0)

    def test_BC_coefMX (self):
        ## Initialize ##
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        xData = df[['internalFactor','externalFactor']]
        bs = Bhattacharyya_Selection(None)
        
        ## Check input ##
        self.assertRaises(ValueError,bs._coefMatrix,"Invalid Input")
        self.assertRaises(ValueError,bs._coefMatrix,{'a':1})
        self.assertRaises(ValueError,bs._coefMatrix,2)

        ## Extract coefficient matrix ##
        coefMx = bs._coefMatrix(xData)
        print(coefMx)
        ans = (len(xData.columns)*len(xData.columns)-len(xData.columns))/2
        self.assertEqual(len(coefMx),ans)

    def test_BC_FeatureSelection (self):
        ## Initialize ##
        df = pd.read_csv(r"C:\Users\USER\Documents\Imperial College London\Summer Module\Dissertation\New Product Forecast\New Product Forecast\Data\qpmSample.csv")
        xData = df
        bs = Bhattacharyya_Selection(3,None)
        
        ## Check input ##
        self.assertRaises(ValueError,bs.FeatureSelection,"Invalid Input")
        self.assertRaises(ValueError,bs.FeatureSelection,{'a':1})
        self.assertRaises(ValueError,bs.FeatureSelection,2)

        ## Feature Selection ##
        res = bs.FeatureSelection(xData)
        self.assertEqual(len(res),3)
        print(res)


class Test_GaussianProcess(unittest.TestCase):
    def test_GP_train(self):
        self.fail()
    def test_GP_predict(self):
        self.fail()
    def test_GP_score(self):
        self.fail()

if __name__ == '__main__':
    unittest.main()
