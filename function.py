"""
This module contains the Function and BestFunction classes.
The BestFunction class is a child class of the Function class.
"""

from typing import NoReturn

import pandas

from iterators import FunctionIterator


class Function:
    """
    This class manipulates the x and y values of a function.
    It also provides convenient methods that make calculation
    of regression easy.
    """

    def __init__(self, name: str) -> NoReturn:
        """
        Arguments:

        name -- the name to be given to the function
        """

        self._name = name
        self.dataframe = pandas.DataFrame()

    def find_y_from_x(self, x) -> str:
        """
        Gets the corresponding y-value from the given x-value using the
        `loc` dataframe method. If not found, an exception is raised

        Arguments:

        x -- x-value for which y-value is to be fetched
        """

        key = self.dataframe["x"] == x

        try:
            return self.dataframe.loc[key].iat[0, 1]
        except IndexError:
            raise IndexError("Unable to fetch y-value")

    @property
    def name(self) -> str:
        """
        This property returns the name of the function
        """

        return self._name

    @classmethod
    def create_from_dataframe(cls, name, dataframe):
        """
        This class method creates a function from a given dataframe.
        When created, the initial column names are overwritten and
        become "x" and "y"
        """

        fxn = cls(name)
        fxn.dataframe = dataframe
        fxn.dataframe.columns = ["x", "y"]

        return fxn

    def __iter__(self):
        return FunctionIterator(self)

    def __sub__(self, other):
        diff = self.dataframe - other.dataframe
        return diff

    def __repr__(self) -> str:
        return f"<Function For {self.name}>"


class BestFunction(Function):
    """
    This class handles the prediction function, regression and training data.

    Note: a tolerance factor must be provided if, for classification purposes,
    tolerance is allowed. If not provided, the maximum deviation between best
    and training function is used.
    """

    def __init__(self, fxn, training_fxn, error) -> NoReturn:
        """
        Arguments:

        fxn -- the best function
        training_fxn -- the data (for training) on which the
        classifying data is based
        error -- the pre-calculated regression
        """

        super().__init__(fxn.name)
        self.dataframe = fxn.dataframe

        self.error = error
        self.training_fxn = training_fxn

        self._tolerance = 1
        self._value_of_tolerance = 1

    def _discover_biggest_deviation(self, best_fxn, training_fxn):
        """
        Given two functions, this method finds the difference and gets
        the biggest from the resulting dataframe.
        """

        distances = training_fxn - best_fxn
        distances["y"] = distances["y"].abs()
        biggest_deviation = max(distances["y"])

        return biggest_deviation

    @property
    def tolerance(self):
        """
        This property returns the tolerance based on the product
        of the tolerance factor and biggest deviation
        """

        self._tolerance = self.tolerance_factor * self.biggest_deviation

        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

    @property
    def tolerance_factor(self):
        """
        This property returns the value of tolerance
        """

        return self._value_of_tolerance

    @tolerance_factor.setter
    def tolerance_factor(self, value):
        self._value_of_tolerance = value

    @property
    def biggest_deviation(self):
        """
        This property returns the biggest deviation between the classifying
        function and the training function on which it's based
        """

        biggest_deviation = self._discover_biggest_deviation(self, self.training_fxn)

        return biggest_deviation


def exponentiated_error(fxn_one, fxn_two):
    """
    Calculates the exponentiated error of another function
    """

    distances = fxn_two - fxn_one
    distances["y"] = distances["y"] ** 2
    deviation_sum = sum(distances["y"])
    return deviation_sum
