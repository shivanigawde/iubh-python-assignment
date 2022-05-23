import pandas as pd
from sqlalchemy import create_engine

class FunctionManager:

    def __init__(self, path_of_csv):
        """
        Parses a local .csv into a list of Functions. On iterating the object, it returns a Function.
        The functions can also be retrieved with the .functions property
        The csv needs a specific structure in which the first column represents x-values and following columns represent y-values
        :param path_of_csv: local path of the csv
        """
        self._functions = []

        #The csv is red by the Panda module and modulated to dataframe
        try:
            self._function_data = pd.read_csv(path_of_csv)
        except FileNotFoundError:
            print("Issue while reading file {}".format(path_of_csv))
            raise

        #The x values are stored and for further process it will fed into each Function
        x_values = self._function_data["x"]

        #The other lines recapitulate over each and every column within panda dataframe which create a new Function object from that data
        for name_of_column, data_of_column in self._function_data.iteritems():
            if "x" in name_of_column:
                continue
            # We previously stored the x column, and now we have the y column. We can combine them together with the concat function
            subset = pd.concat([x_values, data_of_column], axis=1)
            function = Function.from_dataframe(name_of_column, subset)
            self._functions.append(function)


    def to_sql(self, file_name, suffix):
        """
        Writes the data to a local sqlite db using pandas to.sql() method
        If the file already exists, it will be replaced
        :param file_name: the name the db gets
        :param suffix: to comply to the assignment the headers require a specific suffix to the original column name
        """
        #Using SQLalchemy  "engine" is created. It pick up the creation of the db for user, if it is not existent
        engine = create_engine('sqlite:///{}.db'.format(file_name), echo=False)

        # Here I mention functionality from Pandas to write to an sql db.
        # Requirment "engine" object from sqlalchemy
        # We required some specific point to fullfil the assignment's requirment for naming the columns
        # Moreover, the names of the functions are slightly modified to comply
        copy_of_function_data = self._function_data.copy()
        copy_of_function_data.columns = [name.capitalize() + suffix for name in copy_of_function_data.columns]
        copy_of_function_data.set_index(copy_of_function_data.columns[0], inplace=True)

        copy_of_function_data.to_sql(
            file_name,
            engine,
            if_exists="replace",
            index=True,
        )

    @property
    def functions(self):
        """
        Returns a list with all the functions. The user can also just iterate over the object itself.
        :rtype: object
        """
        return self._functions

    def __iter__(self):
        #  It creates object iterable
        return FunctionManagerIterator(self)

    def __repr__(self):
        return "Contains {} number of functions".format(len(self.functions))


class FunctionManagerIterator():

    def __init__(self, function_manager):
        """
        Used for the iteration of a FunctionManager
        :param function_manager:
        """
        #This is the simple class which handles iteration over a FunctionManager
        self._index = 0
        self._function_manager = function_manager

    def __next__(self):
        """
        returns a function object as it iterates over the list of functions
        :rtype: function
        """
        if self._index < len(self._function_manager.functions):
            value_requested = self._function_manager.functions[self._index]
            self._index = self._index + 1
            return value_requested
        raise StopIteration


class Function:

    def __init__(self, name):
        """
        Contains the X and Y values of a function. Underneath it uses a Panda dataframe.
        It has some convenient methods that makes calculating regressions easy.
        1) you can give it a name that can be retrieved later
        2) it is iterable and returns a point represented as dict
        3) you can retrieve a Y-Value by providing an X-Value
        4) you can subtract two functions and get a resulting dataframe with the deviation
        :param name: the name the function should have
        """
        self._name = name
        self.dataframe = pd.DataFrame()

    def locate_y_based_on_x(self, x):
        """
        retrieves a Y-Value
        :param x: the X-Value
        :return: the Y-Value
        """
        # use panda iloc function to identify the x and return the corresponding y
        # If it is not found, an exception is raised
        search_key = self.dataframe["x"] == x
        try:
            return self.dataframe.loc[search_key].iat[0, 1]
        except IndexError:
            raise IndexError


    @property
    def name(self):
        """
        The name of the function
        :return: name as str
        """
        return self._name

    def __iter__(self):
        return FunctionIterator(self)

    def __sub__(self, other):
        """
        Substracts two functions and returns a new dataframe
        :rtype: object
        """
        diff = self.dataframe - other.dataframe
        return diff

    @classmethod
    def from_dataframe(cls, name, dataframe):
        """
        Immediately create a function by providing a dataframe.
        On creation the original column names are overwritten to "x" and "y"
        :rtype: a Function
        """
        function = cls(name)
        function.dataframe = dataframe
        function.dataframe.columns = ["x", "y"]
        return function

    def __repr__(self):
        return "Function for {}".format(self.name)

class IdealFunction(Function):
    def __init__(self, function, training_function, error):
        """
        An ideal function stores the predicting function, training data and the regression.
        Make sure to provide a tolerance_factor if for classification purpose tolerance is allowed
        Otherwise it will default to the maximum deviation between ideal and train function
        :param function: the ideal function
        :param training_function: the training data the classifying data is based upon
        :param squared_error: the beforehand calculated regression
        """
        super().__init__(function.name)
        self.dataframe = function.dataframe

        self.training_function = training_function
        self.error = error
        self._tolerance_value = 1
        self._tolerance = 1

    def _determine_largest_deviation(self, ideal_function, train_function):
        # Take 2 functions and substracts them
        # Result will finds which one is largest
        distances = train_function - ideal_function
        distances["y"] = distances["y"].abs()
        largest_deviation = max(distances["y"])
        return largest_deviation

    @property
    def tolerance(self):
        """
        This property describes the accepted tolerance towards the regression in order to still count as classification.
        Although you can set a tolerance directly (good for unit testing) this is not recommended. Instead provide
        a tolerance_factor
        :return: the tolerance
        """
        self._tolerance = self.tolerance_factor * self.largest_deviation
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):

        self._tolerance = value

    @property
    def tolerance_factor(self):
        """
        Set the factor of the largest_deviation to determine the tolerance
        :return:
        """
        return self._tolerance_value

    @tolerance_factor.setter
    def tolerance_factor(self, value):
        self._tolerance_value = value

    @property
    def largest_deviation(self):
        """
        Retrieves the largest deviation between classifying function and the training function it is based upon
        :return: the largest deviation
        """
        largest_deviation = self._determine_largest_deviation(self, self.training_function)
        return largest_deviation


class FunctionIterator:

    def __init__(self, function):
        #On iterating over a function it gives a dict which describes the point
        self._function = function
        self._index = 0

    def __next__(self):
        # On iterating over a function it gives us the output a dict that describes the point
        if self._index < len(self._function.dataframe):
            value_requested_series = (self._function.dataframe.iloc[self._index])
            point = {"x": value_requested_series.x, "y": value_requested_series.y}
            self._index += 1
            return point
        raise StopIteration
