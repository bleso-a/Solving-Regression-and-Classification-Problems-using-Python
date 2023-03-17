"""
This module contains the Manager class. It is responsible
for parsing a CSV file into a list of functions.
"""

from typing import NoReturn

import pandas
from sqlalchemy import create_engine

from exceptions import ParseError
from function import Function
from iterators import ManagerIterator


class Manager:
    def __init__(self, csv_path: str) -> NoReturn:
        """
        The CSV file must have a format where the first column represents
        x-values and next column represents the y-values

        Here, we read the provided CSV using pandas and convert into a dataframe.
        We then loop over the values in the x-column and create a new Function object.
        While looping, we take care of the values in the y-column and join them using
        the .concat() dataframe method.

        Arguments:
        csv_path -- the path to the CSV path
        """

        self._fxns = []

        try:
            self._fxn_data = pandas.read_csv(csv_path)
        except:
            raise ParseError(f"Could not read {csv_path}. Try again with a valid file!")

        values_in_x_column = self._fxn_data["x"]

        for column_name, column_data in self._fxn_data.items():
            if "x" in column_name:
                continue

            batch = pandas.concat([values_in_x_column, column_data], axis=1)
            self._fxns.append(Function.create_from_dataframe(column_name, batch))

    def write_to_sql(self, file: str, postfix: str):
        """
        Using an engine instance, we create a database and use the .to_sql() from
        pandas to populate it. Also, we are writing a copy of the value of
        `self._fxn_data` because some modifications are required regarding column
        names as stated in the assignment instructions. This way, we preserve the
        integrity of the original data
        """

        engine = create_engine(f"sqlite:///{file}.db", echo=False)
        fxn_data_copy = self._fxn_data.copy()
        fxn_data_copy.columns = [
            name.capitalize() + postfix for name in fxn_data_copy.columns
        ]
        fxn_data_copy.set_index(fxn_data_copy.columns[0], inplace=True)

        print(f"Writing data to {file}.db")
        fxn_data_copy.to_sql(file, engine, if_exists="replace", index=True)
        print(f"Data written to {file}.db successfully")

    @property
    def functions(self) -> list:
        """
        This property returns a list containing all functions
        """

        return self._fxns

    def __repr__(self) -> str:
        return f"<Number Of Functions: {len(self.functions)}>"

    def __iter__(self) -> ManagerIterator:
        return ManagerIterator(self)
