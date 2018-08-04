from GreyBass.Grey_Bass import Grey_Bass as gb
import scipy as spy
import pandas as pd
from pandas import Series

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
        if isinstance(xData,Series):
            xData = xData.tolist()
        if isinstance(yData,Series):
            yData = yData.tolist()

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
    def __init__(self, IMachineLearning: IMachineLearning):
        self._IMachineLearning = IMachineLearning
        self._chromosome= []
        self._populationSize = 0

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
            score = np.sqrt(score/len(actData))
        elif method == 'classificationError':
            for i in range(len(actData)):
                if prdData[i] != actData[i]:
                    score +=1
        return score
    
    def selection (self):
        """
        This function is to randomly select chromosome with the given population size.
        chromosome is a binary set that represent the selected data features.
        To move to the next generation, selection is done from a portion of existing or 
        current population which becomes parent to the next generation.
        """

        return None

    def crossover (self):
        return None

    def mutation (self):
        return None

    def FeatureSelection(self):
        return None

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

