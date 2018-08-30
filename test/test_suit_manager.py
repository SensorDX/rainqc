import unittest
from test.test_pairwiseView import TestPairwiseView

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPairwiseView))


    return suite


if __name__ == '__main__':
    runner =  unittest.TextTestRunner()
    test_suite = suite()
    runner.run(test_suite)
