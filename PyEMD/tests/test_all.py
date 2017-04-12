#import usertest
#import configtest # first test
import unittest   # second test

test_loader = unittest.defaultTestLoader.discover('.')
test_runner = unittest.TextTestRunner()
test_runner.run(test_loader)

# loader = unittest.TestLoader()
# tests = loader.discover('.')
# testRunner = unittest2.runner.TextTestRunner()
# testRunner.run(tests)
