from GreyBass.Grey_Bass import Grey_Bass as gb
import scipy as spy
import pandas as pd
from pandas import Series
import numpy as np
import random
from sklearn.model_selection import KFold
from numpy import ndarray
import math
import scipy.stats as st


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
        self._regression = True

    def train(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def score(self,prd, real):
        ## Check input ##
        if isinstance(prd,Series):
            prd = np.array(prd.tolist())
        if isinstance(real,Series):
            real = np.array(real.tolist())
        if not isinstance(prd,ndarray):
            raise ValueError("Invalid prediction data")
        if not isinstance(real,ndarray):
            raise ValueError("Invalid real data")

        ## Compute Score ##
        score = 0
        if self._regression:
            for i in range(len(prd)):
                score += np.square(prd[i]-real[i])
            score = np.sqrt(score/len(prd))
        else:
            for i in range(len(prd)):
                if prd[i] == real[i]:
                    score += 1
            score = score/len(prd)
        return score

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

from sklearn.svm import SVR, SVC
class SVM_Model (IMachineLearning):
    """
    This class implment SVM regression and classification from Sklearn.
    """

    def __init__(self,regression = True,kernel = 'rbf',gamma ='auto', degree =3, C = 1):
        self._regression = regression
        if regression:
            self._model = SVR(kernel = kernel, degree = degree, gamma = gamma, C = C)
        else:
            self._model = SVC(kernel = kernel, degree = degree, gamma = gamma, C = C)
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
            xData = np.array(xData.tolist())
        ## predict ##
        self._prd = self._model.predict(xData)
        return self._prd


from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
class Gaussian_Model (IMachineLearning):
    """
    This class implment Gaussian process regression and classification from Sklearn.
    """
    def __init__(self,alpha, optimizer= 'fmin_l_bfgs_b', regression = True, kernel = None, n_jobs = 4):
        self._regression = regression
        self._kernel = kernel
        self.alpha = alpha
        self._optimizer = optimizer
        self.njob = n_jobs
        if regression:
            self._model = GaussianProcessRegressor(kernel=kernel,alpha=alpha,optimizer=optimizer)
        else:
            self._model = GaussianProcessClassifier(kernel=kernel,optimizer=optimizer,n_jobs=n_jobs)
        return super().__init__()

    def train(self,xData, yData):
        ## check input ##
        if not isinstance(xData,pd.DataFrame):
            raise ValueError('Invalid xData')

        if not isinstance(yData,pd.DataFrame) and not isinstance(yData,Series):
            raise ValueError('Invalid yData')
        ## train SVM ##
        self._xData = xData
        self._yData = yData
        self._model = self._model.fit(self._xData,self._yData)

    def predict(self,xData):
        ## check input ##
        if isinstance(xData,str):
            raise ValueError('Invalid Argument')
        if not isinstance(xData,pd.DataFrame):
            raise ValueError("Invalid Argument")
        ## predict ##
        self._prd = self._model.predict(xData)
        return self._prd

from sklearn.neighbors import RadiusNeighborsRegressor, RadiusNeighborsClassifier
class Adaptive_KNN_Model (IMachineLearning):
    """
    This class performs Adaptive K-Nearest-Neighbor
    """
    def __init__(self,regression = True, radius = 1.0,weights = 'distance',algorithm = 'auto',leaf_size = 30, p =2,metric = 'minkowski',outlier_label= None,metric_params = None):
        self._regression = regression
        self._radius = radius
        self._weights = weights
        self._algorithm = algorithm
        self._leaf_size = leaf_size
        self._p = p
        self._metric = metric
        self._metric_params =metric_params
        self._outlier_label = outlier_label
        if regression:
            self._model = RadiusNeighborsRegressor(radius,weights,algorithm,leaf_size,p,metric,metric_params)
        else:
            self._model = RadiusNeighborsClassifier(radius,weights,algorithm,leaf_size,p,metric,metric_params)
        return super().__init__()

    def train(self,xData,yData):
        ## check input ##
        if not isinstance(xData,pd.DataFrame):
            raise ValueError('Invalid xData')

        if not isinstance(yData,pd.DataFrame) and not isinstance(yData,Series):
            raise ValueError('Invalid yData')
        ## train SVM ##
        self._xData = xData
        self._yData = yData
        self._model = self._model.fit(self._xData,self._yData)

    def predict(self,xData):
        ## check input ##
        if isinstance(xData,str):
            raise ValueError('Invalid Argument')
        if not isinstance(xData,pd.DataFrame):
            raise ValueError("Invalid Argument")
        ## predict ##
        self._prd = self._model.predict(xData)
        return self._prd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
class Forest_Model (IMachineLearning):
    def __init__(self, regression =True ,n_estimators=10, criterion='gini', 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, max_features='auto', 
                 max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 min_impurity_split=None, bootstrap=True, 
                 oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                 warm_start=False, class_weight=None):

        self._regression = regression
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features =max_features
        self._max_leaf_nodes =max_leaf_nodes
        self._min_impurity_decrease =min_impurity_decrease
        self._min_impurity_split =min_impurity_split
        self._bootstrap =bootstrap
        self._oob_score =oob_score
        self._n_jobs =n_jobs
        self._random_state =random_state
        self._verbose =verbose
        self._warm_start = warm_start
        self._class_weight =class_weight

        if regression:
            self._model = RandomForestRegressor(n_estimators=10, criterion='mse', 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, max_features='auto', 
                 max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 min_impurity_split=None, bootstrap=True, 
                 oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                 warm_start=False)
        else:
            self._model = RandomForestClassifier(n_estimators=10, criterion='gini', 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, max_features='auto', 
                 max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 min_impurity_split=None, bootstrap=True, 
                 oob_score=False, n_jobs=1, random_state=None, verbose=0, 
                 warm_start=False, class_weight=None)
        return super().__init__()

    def train(self,xData,yData):
        ## check input ##
        if not isinstance(xData,pd.DataFrame):
            raise ValueError('Invalid xData')

        if not isinstance(yData,pd.DataFrame) and not isinstance(yData,Series):
            raise ValueError('Invalid yData')
        ## train SVM ##
        self._xData = xData
        self._yData = yData
        self._model = self._model.fit(self._xData,self._yData)

    def predict(self,xData):
        ## check input ##
        if isinstance(xData,str):
            raise ValueError('Invalid Argument')
        if not isinstance(xData,pd.DataFrame):
            raise ValueError("Invalid Argument")
        ## predict ##
        self._prd = self._model.predict(xData)
        return self._prd

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
        self._bestSelection = {'fitness':0,'selection':None}



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
            score = 1/np.sqrt(score/len(actData))
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

        ## store best selection ##
        if max(fitnessList) > self._bestSelection['fitness']:
            self._bestSelection['fitness'] = max(fitnessList)
            colIx = np.where(population[fitnessList.index(max(fitnessList))]==1)
            self._bestSelection['selection'] = xData[xData.columns[colIx]].columns
            ## store elite list which goes to next generation directly ##
            self._elite.append(population[fitnessList.index(max(fitnessList))])

        ## return current generation selection and used for crossover ##
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
            print(round(i/iter*100,2),"%")
            parents = self.selection(xData,yData,population,self._selectedNum,fitnessMethod)
            if i < iter:
                nextGen = self.crossover(parents)
                population = self.mutation(nextGen)
        return self._bestSelection

class Bhattacharyya_Selection (IFeatureSelection):
    def __init__(self,selectNum =20, IMachineLearning=None):
        self._coefFrame = pd.DataFrame(columns=['distA','distB','coef'])
        self._selectFeature = []
        self._selectNum = selectNum
        return super().__init__(IMachineLearning)

    def Bhattacharyya_coefficient(self,distA:ndarray, distB:ndarray):
        
        ## check input ##
        if isinstance(distA,Series):
            distA = np.array(distA.tolist())
        if isinstance(distB,Series):
            distB = np.array(distB.tolist())
        if not isinstance(distA,ndarray) and not isinstance(distB,ndarray):
            raise ValueError("Invalid Distributions")
        if len(distA) != len(distB):
            raise ValueError("distributions A and B should have same length")

        ## Bhattacharyya coefficient ##
        distA = self._standardize(distA)
        distB = self._standardize(distB)

        coef = 0
        for i in range(len(distA)):
            temp = math.sqrt(distA[i] * distB[i])
            coef = coef + temp
        return coef

    def _standardize(self, dist: ndarray):

        ## Check input ##
        if isinstance(dist,Series):
            dist = np.array(dist.tolist())
        if not isinstance(dist,ndarray):
            raise ValueError('Invalid distribution')
        ## scale distribution ##
        res = []
        max = np.max(dist)
        min = np.min(dist)
        dif = max - min
        dist = np.subtract(dist,min)
        dist = np.divide(dist,dif)
        dist = np.sort(dist)
        return dist

    def FeatureSelection (self, xData:pd.DataFrame):
        """
        Larger the coefficient, larger the overlap between two variables.
        The selection process is to select the most divergent vairable pairs.
        """
        ## Check input ##
        if not isinstance(xData,pd.DataFrame):
            raise ValueError("Invalid xData")

        ## Feature Selection ##
        coefMx = self._coefMatrix(xData)
        coefMx = coefMx.sort_values(["coef"])
        index = 0
        self._selectFeature = []
        while len(self._selectFeature) < self._selectNum:
            self._selectFeature.append(coefMx.iloc[index].distA)
            self._selectFeature.append(coefMx.iloc[index].distB)
            self._selectFeature = list(set(self._selectFeature))
            index += 1
        if len(self._selectFeature) > self._selectNum:
            self._selectFeature.pop()

        return self._selectFeature

    def _coefMatrix (self, xData):
        """
        This function iterate through all xData and create a 
        Bhattacharyya coefficient matrix.
        """
        ## check input ##
        if not isinstance(xData,pd.DataFrame):
            raise ValueError("Invalid input")

        ## Extract matrix ##
        cols = xData.columns
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i < j:
                    coef = self.Bhattacharyya_coefficient(xData[cols[i]],xData[cols[j]])
                    self._coefFrame=self._coefFrame.append({'distA':cols[i],'distB':cols[j],'coef':coef}, ignore_index = True)     
        return self._coefFrame


