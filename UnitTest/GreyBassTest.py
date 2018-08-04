import unittest
from GreyBass import Grey_Bass
import numpy as np
import numpy.testing as testing
import pandas as pd

class Test_GreyBassTest(unittest.TestCase):

    def test_meanSeq(self):
        pdData = pd.DataFrame({'data':[1,2,3,4,5]})
        gb = Grey_Bass.Grey_Bass()
        testData = np.array([1,2,3,4,5])
        res = gb._meanSeq(testData)
        res = gb._meanSeq(pdData.data)
        ans = np.array([2,4.5,8,12.5])
        print(res)
        print(ans)
        testing.assert_array_almost_equal(ans,res)
        self.assertRaises(ValueError,gb._meanSeq,1)
        self.assertRaises(ValueError,gb._meanSeq,[1])
        self.assertRaises(ValueError,gb._meanSeq,'Not Valid')
        
    def test_agoSeq(self):
        pdData = pd.DataFrame({'data':[1,2,3,4,5]})
        gb = Grey_Bass.Grey_Bass()
        testData = np.array([1,2,3,4,5])
        res = gb._agoSeq(testData)
        res = gb._agoSeq(pdData.data)
        ans = np.array([1,3,6,10,15])
        print(res)
        print(ans)
        testing.assert_array_almost_equal(ans,res)
        self.assertRaises(ValueError,gb._agoSeq,1)
        self.assertRaises(ValueError,gb._agoSeq,'Not Valid')
        
    def test_NLS(self):
        gb = Grey_Bass.Grey_Bass()
        testData = np.arange(1000,5000,200)
        pdData = pd.DataFrame({'data':[1,2,3,4,5]})
        res = gb._NLS(testData)
        print(res.x)
        res = gb._NLS(pdData.data)
        self.assertNotAlmostEqual(gb._internalFactor,0)
        self.assertNotAlmostEqual(gb._externalFactor,0)
        self.assertGreater(gb._marketSize,0)
        self.assertRaises(ValueError,gb._NLS,1)
        self.assertRaises(ValueError,gb._NLS,[1])
        self.assertRaises(ValueError,gb._NLS,'Not Valid')
        

    def test_predict(self):
        pdData = pd.DataFrame({'data':[1,2,3,4,5]})
        gb = Grey_Bass.Grey_Bass()
        trainData = np.arange(100,200,10)
        gb._NLS(trainData)
        testData = np.arange(201,300,10)
        gb.predict(pdData.data)
        res = gb.predict(testData, False)
        for i in res:
            self.assertNotEqual(i,0)
        score = 0
        for i in range(len(res)):
            temp = np.square(res[i] - testData[i])
            score += temp
        score = np.sqrt(score/len(res))
        print(res)
        print(score)
        self.assertRaises(ValueError,gb.predict,'Not Valid')

    def test_whitenisation (self):
        pdData = pd.DataFrame({'data':[1,2,3,4,5]})
        gb = Grey_Bass.Grey_Bass()
        trainData = np.arange(100,200,1)
        gb._NLS(trainData)
        testData = np.arange(201,300,10)
        res = gb.predict(testData, True)
        gb.predict(pdData.data,True)
        for i in res:
            self.assertNotEqual(i,0)
        score = 0
        for i in range(len(res)):
            temp = np.square(res[i] - testData[i])
            score += temp
        score = np.sqrt(score/len(res))
        print(res)
        print(score)

    def test_rmse(self):
        gb = Grey_Bass.Grey_Bass()
        resData = np.arange(100,200,1)
        realData = np.arange(101,201,1)
        res = gb.rmse(resData,realData)
        self.assertRaises(ValueError,gb.rmse,[1,1],[1])


if __name__ == '__main__':
    unittest.main()
