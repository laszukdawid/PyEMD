import logging
import unittest
import sys

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_suite = unittest.defaultTestLoader.discover('.', '*test*.py')
    test_runner = unittest.TextTestRunner(resultclass=unittest.TextTestResult)
    result = test_runner.run(test_suite)
    sys.exit(not result.wasSuccessful())
