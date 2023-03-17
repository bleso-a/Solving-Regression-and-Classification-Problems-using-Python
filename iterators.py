"""
This module contains the ManagerIterator and FunctionIterator classes.
"""

from typing import NoReturn


class ManagerIterator:
    """
    This class enables us iterate over a Manager instance
    """

    def __init__(self, fxn_manager) -> NoReturn:
        self._pointer = 0
        self._fxn_manager = fxn_manager

    def __next__(self):
        if self._pointer < len(self._fxn_manager.functions):
            next_item = self._fxn_manager.functions[self._pointer]
            self._pointer = self._pointer + 1
            return next_item

        raise StopIteration


class FunctionIterator:
    """
    This class enables us iterate over a Function instance.
    It returns a dictionary with details of a particular point.
    """

    def __init__(self, fxn) -> NoReturn:
        self._fxn = fxn
        self._pointer = 0

    def __next__(self):
        if self._pointer < len(self._fxn.dataframe):
            next_item = self._fxn.dataframe.iloc[self._pointer]
            point = {"x": next_item.x, "y": next_item.y}
            self._pointer = self._pointer + 1
            return point

        raise StopIteration
