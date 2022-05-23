from unittest import TestCase
import pandas as pd
from function import Function
from lossfunction import squared_error

class Test(TestCase):
    def setUp(self):
        # Setting up some functions
        data1 = {"x":[1.0,2.0,3.0],"y":[5.0,6.0,7.0]}
        self.dataframe1 = pd.DataFrame(data=data1)

        data2 = {"x":[1.0,2.0,3.0], "y":[7.0,8.0,9.0]}
        self.dataframe2 = pd.DataFrame(data=data2)

        self.function1 = Function("name")
        self.function1.dataframe = self.dataframe1

        self.function2 = Function("name")
        self.function2.dataframe = self.dataframe2


    def tearDown(self):
        pass

    def test_squared_error(self):
        # case 1: simple test if loss function computes correct value
        self.assertEqual(squared_error(self.function1, self.function2), 12.0)
        # case 2: simple test if loss function is associative
        self.assertEqual(squared_error(self.function2, self.function1), 12.0)
        # case 3: check if regression of two equal functions is 0
        self.assertEqual(squared_error(self.function1, self.function1), 0.0)


