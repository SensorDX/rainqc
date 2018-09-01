import unittest
from test.test_pairwiseView import TestPairwiseView
from test.test_RQC import TestRQC
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPairwiseView))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRQC))

    return suite


if __name__ == '__main__':
    runner =  unittest.TextTestRunner()
    test_suite = suite()
    runner.run(test_suite)
