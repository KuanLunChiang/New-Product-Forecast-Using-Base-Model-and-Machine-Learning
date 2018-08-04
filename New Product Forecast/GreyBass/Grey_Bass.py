import numpy as np
import scipy as sp
from scipy.optimize import least_squares
from pandas import Series

class Grey_Bass(object):

    """
    This Class generate the Grey Bass Model. 
    The key attributes include external adaptor, internal imitator, and market size.
    """
    def __init__(self):
        self._externalFactor = 0
        self._internalFactor = 0
        self._marketSize = 0
        self._baseSeq = 0
    
    def _meanSeq(self,input:list):
        '''
        this is a support function which create the mean sequence by consecutive neighbors
        '''
        ## Input Check ##
        if isinstance(input,str):
            raise ValueError('Invalid Input')
        if not isinstance(input,list):
            try:
                input = list(input)
            except:
                raise ValueError('Invalid Input')
        if len(input)<2:
            raise ValueError('Input length should be greater than 1')  
        
        ## Compute mean sequence ##
        res = []
        input = self._agoSeq(input)
        for i in range(len(input)-1):
            temp = (input[i]+input[i+1])*0.5
            res.append(temp)
        res = np.array(res)
        return res


    def _agoSeq(self,input:list):
        '''
        This function is used for generating accumulating sequence
        '''
        ## Input check ##
        if isinstance(input,str):
            raise ValueError('Invalid Input')
        if not isinstance(input,list):
            try:
                input = list(input)
            except:
                raise ValueError('Invalid Input')

        ## Accumulate generate sequence ##
        res = []
        for i in range(len(input)):
            if i == 0:
                res.append(input[i])
            else:
                res.append(res[i-1]+input[i])
        return res


    def _NLS(self, input:list):
        '''
        This function implment nonlinear least square to compute external, internal, and market size.
        '''
        ## Input check ##
        if isinstance(input,Series):
            input = input.tolist()
        if isinstance(input,str):
            raise ValueError('Invalid Input')
        if not isinstance(input,list):
            try:
                input = list(input)
            except:
                raise ValueError('Invalid Input')
        if len(input)<2:
            raise ValueError('Input length should be greater than 1')  

        ## Nonlinear least square ##
        paraList = [0.1,0.1,100]
        self._baseSeq = input
        res = least_squares(self.__objFunction,paraList,method = 'lm')
        self._externalFactor = res.x[0]
        self._internalFactor = res.x[1]
        self._marketSize = res.x[2]
        return res

    def __objFunction(self,paraList:list):
        '''
        This return the loss function and used in nonlinear least squares
        '''
        input = self._baseSeq
        res = []
        a = paraList[0]
        b = paraList[1]
        m = paraList[2]
        z = self._meanSeq(input)
        for i in range(len(input)-1):
            res.append(np.square(input[i+1] + (a-b)*z[i]- (a * m) + (b/m *(np.square(z[i])))))
        return res

    def _whitenisation(self,input:list):
        '''
        this function is used for withen the grey differential equation.
        '''

        ## Input check ##
        if isinstance(input,str):
            raise ValueError('Invalid Input')
        if not isinstance(input,list):
            try:
                input = list(input)
            except:
                raise ValueError('Invalid Input')

        ## Whitenisation ##
        a = self._externalFactor
        b = self._internalFactor
        m = self._marketSize
        x1List = []
        zList = []
        for i in range(len(input)):
            tempow = -1*(a+b)*input[i]
            temp = m*((1-np.exp(tempow)/(1+(b/a*np.exp(tempow)))))
            x1List.append(temp)
            if i > 0:
                zList.append((x1List[i]+x1List[i-1])*0.5)
        return x1List, zList

    def predict(self,input:list, whiten = False):
        '''
        This function will return the prediction value 
        given with the NLS external, internal, and market size
        '''

        ## Input check ##
        if isinstance(input,Series):
            input = input.tolist()
        if isinstance(input,str):
            raise ValueError('Invalid Input')
        if not isinstance(input,list):
            try:
                input = list(input)
            except:
                raise ValueError('Invalid Input')
        
        ## Prediction ##
        a = self._externalFactor
        b = self._internalFactor
        m = self._marketSize
        if whiten == False:
            x1 = self._agoSeq(input)
            z = self._meanSeq(input)
        else:
            x1,z = self._whitenisation(input)
        resList = []
        for i in range(len(input)):
            res = a*(m-x1[i])+b*x1[i]*(1-(x1[i]/m))
            #res = a*(m-z[i])+b*z[i]*(1-(z[i]/m))
            #res = -(a-b)*z[i]+a*m-(b/m*np.square(z[i]))
            resList.append(res)
        return resList


    def rmse(self,prd:list, real:list):
        '''
        This Function compute the rmse score
        '''
        ## Input check ##
        if not isinstance(prd,list):
            try:
                prd = list(prd)
            except:
                raise ValueError('Invalid Prdiction list')
        if not isinstance(real,list):
            try:
                prd = list(real)
            except:
                raise ValueError('Invalid Real Value list')
        if len(prd) != len(real):
            raise ValueError('Different Length')

        ## Square root mean sum of square error ##
        score = 0
        for i in range(len(prd)):
            score += np.square(prd[i]-real[i])
        score = np.sqrt(score/len(prd))
        return score