import unittest
#from src import TestPairwiseView
from test_tahmoDataSource import TestTahmoDataSource

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestTahmoDataSource))
    #suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPairwiseView))
    #suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRQC))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run(test_suite)
