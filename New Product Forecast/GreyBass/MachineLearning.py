from GreyBass.Grey_Bass import Grey_Bass as gb
import scipy as spy
import pandas as pd
from pandas import Series
import numpy as np
import random
from sklearn.model_selection import KFold
from numpy import ndarray

#### Interfaces ####
class IMachineLearning(object):
    """
    This intreface is used to create a standard input and output 
    for new product competitability forecast using Bass model and machine learning
    """
    def __init__(self):
        self._xData = []
        self._yData = []
        self._prd =[]

    def train(self):
        raise NotImplementedError()
    def predict(self):
        raise NotImplementedError()
    def score(self):
        raise NotImplementedError()


class IBass(object):
    def __init__(self):
        self._internalFactor = {}
        self._externalFactor = {}
        self._marketSize = {}
        self._prdInternal = {}
        self._prdExternal = {}
        self._prdMarket = {}

class IFeatureSelection (object):

    '''
    Feature selection interface is going to 
    provide the standard input and output for feature selection algorithms
    '''
    def __init__(self, IMachineLearning):
        self._IMachineLearning = IMachineLearning
        self._selectFeature = {}
        self._xData = []
        self._yData = []
    def FeatureSelection (self):
        raise NotImplmentError()
    

### Classes ###
from GreyBass.Grey_Bass import Grey_Bass
class BassInformation (IBass):
    """
    This class is used to derive the internal factor, external factor, and market size
    from Grey Bass Model.
    """
    def __init__(self):
        return super().__init__()

    def train_NLS (self,xData:dict):
        """
        This function is to train the Grey Bass Model
        and derived the internal factor, external factor, 
        and market size. The output is in the List format.
        Parameters:
            xData: dictionary type or panda Frame. 
                   The keys are the response variable, 
                   and the values are the sales data.
        Output:
            _internalFactor, _externalFactor, _marketSize:
                these three outputs are dictionary type.
                The keys are the response variable.
                These variables are used for train_ML.
        """
        ## Input check ##
        if not isinstance(xData,dict) and not isinstance(xData,pd.DataFrame):
            raise ValueError('Invalid xData')
        if isinstance(xData,pd.DataFrame):
            xData = xData.to_dict()

        ## Train Bass Model ##
        for i in xData:
            gb = Grey_Bass()
            gb._NLS(xData[i])
            self._externalFactor[i] = gb._externalFactor
            self._internalFactor[i] = gb._internalFactor
            self._marketSize[i] = gb._marketSize

class Bass_prd (BassInformation,IMachineLearning):
    """
    This class inhereted from BassInformation and implment the IMachineLearning.
    The main purpose of this class is to predict the internal factor, 
    external factor, and market size by machine learning model implementing IMachineLearning.
    """
    def __init__(self):
        return super().__init__()

    def train (self, model: IMachineLearning, xData:dict, yData: dict):
        return None

    def predict(self):
        return super().predict()

from sklearn.svm import SVR
class SVM_Model (IMachineLearning):

    def __init__(self,kernel = 'rbf',gamma ='auto', degree =3, C = 1):
        self._model = SVR(kernel = kernel, degree = degree, gamma = gamma, C = C)
        self._kernel = kernel
        self._gamma = gamma
        self._degree = degree
        self._C = C
        return super().__init__()
    
    def train(self,xData, yData):
        ## check input ##
        if isinstance(xData,str) or isinstance(yData,str):
            raise ValueError('Invalid Argument')

        ## train SVM ##
        self._xData = xData
        self._yData = yData
        self._model = self._model.fit(self._xData,self._yData)
        
    def predict(self,xData):
        ## check input ##
        if isinstance(xData,str):
            raise ValueError('Invalid Argument')
        if isinstance(xData,Series):
            xData = xData.tolist()
        ## predict ##
        self._prd = self._model.predict(xData)
        return self._prd

    def score(self):

        return super().score()

class Gaussian_Model (IMachineLearning):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

class Deep_Learning_Model (IMachineLearning):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

class Forest_Model (IMachineLearning):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

class Genetic_Selection (IFeatureSelection):
    """
    This class implments Genetic Algorithm for feature selection.
    """
    def __init__(self, model: IMachineLearning,selectedNum:int ,populationSize = 100, mutateRate = 0.3):
        ## Check input ##
        if not isinstance(populationSize,int):
            raise ValueError("Invalid population size")
        if not isinstance(model,IMachineLearning):
            raise ValueError("Invalid model")
        if not isinstance(selectedNum,int):
            raise ValueError("Invalid select number")

        ## Initialize class ##
        self._IMachineLearning = model
        self._populationSize = populationSize
        self._population = None
        self._elite = []
        self._selectedNum = selectedNum
        self._mutateRate = mutateRate
        self._bestSelection = None



    def fitness (self, prdData, actData, method = 'rmse'):
        """
        There are two method provided for fitness function: RMSE or Classification error.
        RMSE is used for regression type model, while classification error is for classification model.
        This is the most crucial function for Genetic Algorithm. It determine the termination point and selection weight.
        """
        ## Check input ##
        if not isinstance(method,str):
            raise ValueError("Invalid method")
        if method not in ['rmse','classificationError']:
            raise ValueError("Invalid method")
        if isinstance(prdData,list):
            prdData = list(prdData)
        if isinstance(actData,list):
            actData = list(actData)
        if len(actData) != len(prdData):
            raise ValueError("actData and prdData should have the same length")

        ## Fitness Score ##
        score = 0
        if method == 'rmse':
            for i in range(len(actData)):
                score += np.square(prdData[i] - actData[i])
            score = 1/score/len(actData)
        elif method == 'classificationError':
            for i in range(len(actData)):
                if prdData[i] != actData[i]:
                    score +=1
            score = 1-score/len(actData)
        return score
    
    def _generatePopulation(self,featureNum:int):
        ## check input ##
        if not isinstance(featureNum,int):
            raise ValueError("Invalid featureNum")
        population = []
        for i in range(self._populationSize):
            idx = np.random.choice(np.arange(0,featureNum),size = self._selectedNum,replace = False)
            chromosome = np.zeros(featureNum)
            for index in idx:
                chromosome[index] = 1
            population.append(chromosome)
        self._population = population
        return population

    def selection (self, xData, yData, population:list, selectNum:int,fitnessMethod='rmse'):
        """
        This function is to randomly select chromosome with the given population size.
        chromosome is a binary set that represent the selected data features.
        To move to the next generation, selection is done from a portion of existing or 
        current population which becomes parent to the next generation.
        """

        ## check input ##
        if not isinstance(population,list):
            raise ValueError("Invalid population")
        if not isinstance(selectNum,int):
            raise ValueError("Invalid select number")
        if not isinstance(xData,pd.DataFrame):
            raise ValueError("xData should be Pandas Frame")        
        if not isinstance(yData,pd.DataFrame) and not isinstance(yData,Series):
            raise ValueError("yData should be Pandas Frame")

        ## compute fitness ##
        fitnessList = []
        kf = KFold(5)
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for train_index, test_index in kf.split(xData):
            x_train.append(xData.iloc[train_index])
            x_test.append(xData.iloc[test_index])
            y_train.append(yData.iloc[train_index])
            y_test.append(yData.iloc[test_index])

        for i in population:
            i = np.array(i)
            colIx = np.where(i == 1)
            cvScore = 0
            for j in range(5):
                trainx = x_train[j][x_train[j].columns[colIx]]
                trainy = y_train[j]
                testx = x_test[j][x_test[j].columns[colIx]]
                self._IMachineLearning.train(trainx,trainy)
                prd = self._IMachineLearning.predict(testx)
                cvScore += self.fitness(prd,y_test[j].tolist(),fitnessMethod)
            cvScore = cvScore/5
            fitnessList.append(cvScore)
        self._elite.append(population[fitnessList.index(max(fitnessList))])
        pop = np.arange(0,len(population))
        selectionIX = random.choices(pop,k = selectNum,weights = fitnessList)
        selection = np.array(population)[selectionIX]
        return selection

    def crossover (self, parentId:ndarray):
        """
        Combine two parents chromosome to create a child.
        The parent with the best fitness is untouched and added onto next generation.
        The elite chromosome always goes to the new generation.
        """
        ## check input ##
        if isinstance(parentId,str):
            raise ValueError("Invalid parentId")
        if not isinstance(parentId,ndarray):
            raise ValueError("Invalid parentId")
        ## crossover ##
        nextGeneration = []
        for i in range(len(parentId)-1):
            childA,childB = self._breed(parentId[i],parentId[i+1])
            nextGeneration.append(childA)
            nextGeneration.append(childB)
        nextGeneration = np.array(nextGeneration)
        return nextGeneration

    def _breed(self, parentA:ndarray, parentB:ndarray):
        ## check input ##
        if not isinstance(parentA,np.ndarray):
            raise ValueError("Invalid parentA")
        if not isinstance(parentB,np.ndarray):
            raise ValueError("Invalid parentB")

        ## breed ##
        childAIX = []
        childBIX = []
        IxA = np.where(parentA == 1)
        IxB = np.where(parentB == 1)
        for i in range(len(IxA)):
            if isinstance(len(IxA)/2,float):
                id = (len(IxA)/2)-1
            else:
                id = (len(IxA)+1)/2
            childAIX.append(IxA[i])
            childBIX.append(IxB[len(IxA)-i-1])
            if i == id:
                break
        childA = np.zeros(len(parentA))
        childB = np.zeros(len(parentB))
        childA[childAIX] = 1
        childB[childBIX] = 1
        return childA,childB

    def mutation (self,nextG: ndarray):
        """
        Randomly create a new generation to improve the diversification.
        """

        ## check input ##
        if not isinstance(nextG,ndarray):
            raise ValueError("Invalid nextG")

        ## mutate ##
        nextGeneration = []
        mutateIx = np.arange(0,len(nextG),1)
        mutateNum = int(len(nextG)*self._mutateRate)
        IxM = np.random.choice(mutateIx,size = mutateNum,replace = False)
        for i in range(len(nextG)):
            if i in IxM:
                temp = self._randomSwitch(nextG[i])
                nextGeneration.append(temp)
            else:
                nextGeneration.append(nextG[i])
        for i in self._elite:
            nextGeneration.append(i)
        return nextGeneration

    def _randomSwitch (self, inputList: ndarray):
        if not isinstance(inputList,ndarray):
            raise ValueError("Invalid input")
        IxZ = np.where(inputList == 0)
        IxZ = np.array(IxZ)[0]
        IxS = np.random.choice(IxZ,int(sum(inputList)),False)
        switchList = np.zeros(len(inputList))
        switchList[IxS] = 1
        return switchList
        
    def FeatureSelection(self,xData,yData,fitnessMethod = 'rmse',iter = 1000):
        """
        The main function for Genetic Algorithm. 
        It will terminate when iteration number is reached.
        """
        ## Genetic Algorithm ##
        population = self._generatePopulation(len(xData.columns))
        for i in range(iter):
            parents = self.selection(xData,yData,population,self._selectedNum)
            if i < iter:
                nextGen = self.crossover(parents)
                population = self.mutation(nextGen)

        ## find the best feature selection ##
        fitnessList = []
        kf = KFold(5)
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for train_index, test_index in kf.split(xData):
            x_train.append(xData.iloc[train_index])
            x_test.append(xData.iloc[test_index])
            y_train.append(yData.iloc[train_index])
            y_test.append(yData.iloc[test_index])

        for i in self._elite:
            i = np.array(i)
            colIx = np.where(i == 1)
            print(colIx)
            cvScore = 0
            for j in range(5):
                trainx = x_train[j][x_train[j].columns[colIx]]
                trainy = y_train[j]
                testx = x_test[j][x_test[j].columns[colIx]]
                self._IMachineLearning.train(trainx,trainy)
                prd = self._IMachineLearning.predict(testx)
                cvScore += self.fitness(prd,y_test[j].tolist(),fitnessMethod)
            cvScore = cvScore/5
            fitnessList.append(cvScore)
        print(fitnessList)
        Ixd = self._elite[fitnessList.index(max(fitnessList))]
        self._bestSelection = xData.columns[np.where(Ixd == 1)]
        return self._bestSelection

class Dimension_Selection (IFeatureSelection):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def FeatureSelection(self):
        return super().FeatureSelection()

class Bhattacharyya_Selection (IFeatureSelection):
    def __init__(self, IMachineLearning):
        return super().__init__(IMachineLearning)

    def FeatureSelection(self):
        return super().FeatureSelection()

