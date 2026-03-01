import typing

import numpy
import pandas


class Sample:
    @typing.overload
    def __init__(self, X: pandas.Series, bool_columns: typing.List[str], numeric_columns: typing.List[str]): ...

    @typing.overload
    def __init__(self, binary: numpy.ndarray, numeric: numpy.ndarray): ...

    def __init__(self, *args):
        if len(args) == 3:
            X, bool_columns, numeric_columns = args
            self.numeric = X[numeric_columns].to_numpy().astype(numpy.float64)
            self.binary = X[bool_columns].to_numpy().astype(bool)
        elif len(args) == 2:
            binary, numeric = args
            self.binary = binary
            self.numeric = numeric
        else:
            assert False, "args must have 2 or 3 items (see overloads)"


class Subset:
    def __init__(self, X: pandas.DataFrame, bool_columns: typing.List[str], numeric_columns: typing.List[str]):
        self.binary = X[bool_columns].to_numpy().astype(bool)
        self.numeric = X[numeric_columns].to_numpy().astype(numpy.float64)

    def __iter__(self):
        for binary, numeric in zip(self.binary, self.numeric):
            yield Sample(binary, numeric)

    def __len__(self):
        return len(self.binary)


class Dataset:
    def __init__(self, X: pandas.DataFrame, y: pandas.Series):
        self.bool_columns = list(X.columns[X.dtypes == "bool"])
        self.numeric_columns = list(X.columns[X.dtypes != "bool"])
        self.positive = Subset(X[y == 1], self.bool_columns, self.numeric_columns)
        self.negative = Subset(X[y == 0], self.bool_columns, self.numeric_columns)

    def make_sample(self, X: pandas.Series) -> Sample:
        return Sample(X, self.bool_columns, self.numeric_columns)
