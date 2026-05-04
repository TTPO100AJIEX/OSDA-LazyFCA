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
            self.numeric: numpy.ndarray = X[numeric_columns].to_numpy().astype(numpy.float64)
            self.binary: numpy.ndarray = X[bool_columns].to_numpy().astype(bool)
        elif len(args) == 2:
            binary, numeric = args
            self.binary: numpy.ndarray = binary
            self.numeric: numpy.ndarray = numeric
        else:
            assert False, "args must have 2 or 3 items (see overloads)"


class Subset:
    @typing.overload
    def __init__(self, X: pandas.DataFrame, bool_columns: typing.List[str], numeric_columns: typing.List[str]): ...

    @typing.overload
    def __init__(self, binary: numpy.ndarray, numeric: numpy.ndarray): ...

    def __init__(self, *args):
        if len(args) == 3:
            X, bool_columns, numeric_columns = args
            self.binary = numpy.ascontiguousarray(X[bool_columns].to_numpy().astype(bool))
            self.numeric = numpy.ascontiguousarray(X[numeric_columns].to_numpy().astype(numpy.float64))
        elif len(args) == 2:
            binary, numeric = args
            self.binary = numpy.ascontiguousarray(binary)
            self.numeric = numpy.ascontiguousarray(numeric)
        else:
            assert False, "args must have 2 or 3 items (see overloads)"

    def __iter__(self):
        for binary, numeric in zip(self.binary, self.numeric):
            yield Sample(binary, numeric)

    def __len__(self):
        return len(self.binary)

    def __getitem__(self, i: int):
        return Sample(self.binary[i], self.numeric[i])

    @staticmethod
    def concatenate(subsets: typing.Sequence["Subset"]) -> "Subset":
        """Concatenate several subsets into a single one (used to build the
        opposers of a given class in the multi-class setting)."""
        if not subsets:
            return Subset(
                numpy.empty((0, 0), dtype=bool),
                numpy.empty((0, 0), dtype=numpy.float64),
            )
        binary = numpy.concatenate([s.binary for s in subsets], axis=0)
        numeric = numpy.concatenate([s.numeric for s in subsets], axis=0)
        return Subset(binary, numeric)


class Dataset:
    """Dataset for multi-class LazyFCA.

    Class labels are expected to be integers in the range ``[0, n_classes)``.
    For backward compatibility with the original binary implementation the
    properties :attr:`positive` and :attr:`negative` are exposed when there are
    exactly two classes (mapping to class ``1`` and class ``0`` respectively).
    """

    def __init__(
        self,
        X: pandas.DataFrame,
        y: pandas.Series,
        n_classes: typing.Optional[int] = None,
    ):
        self.bool_columns = list(X.columns[X.dtypes == "bool"])
        self.numeric_columns = list(X.columns[X.dtypes != "bool"])

        y_array = numpy.asarray(y)
        if n_classes is None:
            unique_labels = numpy.unique(y_array)
            inferred = int(unique_labels.max()) + 1 if len(unique_labels) > 0 else 0
            n_classes = max(int(inferred), int(len(unique_labels)))
        self.n_classes = int(n_classes)
        assert self.n_classes >= 2, f"n_classes must be >= 2, got {self.n_classes}"

        self.subsets: typing.List[Subset] = [
            Subset(X[y_array == c], self.bool_columns, self.numeric_columns) for c in range(self.n_classes)
        ]
        # Pre-compute the opposers (samples of all other classes) for every class.
        self.opposers: typing.List[Subset] = [
            Subset.concatenate([self.subsets[j] for j in range(self.n_classes) if j != c])
            for c in range(self.n_classes)
        ]

        self.binary_feature_count = len(self.bool_columns)
        self.numeric_feature_count = len(self.numeric_columns)

        if self.numeric_columns:
            numeric = X[self.numeric_columns].to_numpy().astype(numpy.float64)
            self.numeric_minimum = numeric.min(axis=0)
            self.numeric_maximum = numeric.max(axis=0)
            self.numeric_range = self.numeric_maximum - self.numeric_minimum
        else:
            self.numeric_minimum = numpy.array([], dtype=numpy.float64)
            self.numeric_maximum = numpy.array([], dtype=numpy.float64)
            self.numeric_range = numpy.array([], dtype=numpy.float64)

    @property
    def positive(self) -> Subset:
        """Backward-compat alias: positive samples = class label ``1``."""
        assert self.n_classes == 2, ".positive is only defined for binary datasets (n_classes == 2)"
        return self.subsets[1]

    @property
    def negative(self) -> Subset:
        """Backward-compat alias: negative samples = class label ``0``."""
        assert self.n_classes == 2, ".negative is only defined for binary datasets (n_classes == 2)"
        return self.subsets[0]

    def supporters_of(self, class_index: int) -> Subset:
        return self.subsets[class_index]

    def opposers_of(self, class_index: int) -> Subset:
        return self.opposers[class_index]

    def make_sample(self, X: pandas.Series) -> Sample:
        return Sample(X, self.bool_columns, self.numeric_columns)
