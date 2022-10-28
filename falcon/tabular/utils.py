from copy import deepcopy
import pandas as pd
from typing import Union, Tuple, Optional, List
import numpy as np
from numpy import isin, typing as npt
from falcon import types as ft
from ..abstract.task_pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import balanced_accuracy_score, r2_score


def read_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".parquet"):
        data = pd.read_parquet(path)
    else:
        raise ValueError("Only `.csv` and `.parquet` files are supported")

    return data


def clean_data(
    data: Union[pd.DataFrame, npt.NDArray]
) -> Union[pd.DataFrame, npt.NDArray]:
    if isinstance(data, pd.DataFrame):
        return data.dropna()
    else:
        mask = pd.isnull(data)
        keep = []
        for row in mask:
            if True in row:
                keep.append(False)
            else:
                keep.append(True)
        data = data[keep, :]
        return data


def clean_data_split(X: npt.NDArray, y: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    mask_x = pd.isnull(X)
    mask_y = pd.isnull(y)
    keep = []
    for i in range(len(mask_x)):
        if len(y.shape) == 1:
            if True not in mask_x[i] and mask_y[i] == False:
                keep.append(True)
            else:
                keep.append(False)
        else:
            if True not in mask_x[i] and True not in mask_y[i]:
                keep.append(True)
            else:
                keep.append(False)
    X = X[keep, :]
    if len(y.shape) == 1:
        y = y[keep]
    else:
        y = y[keep, :]
    return X, y


def convert_to_np_obj(
    data: Union[pd.DataFrame, npt.NDArray]
) -> npt.NDArray[np.object_]:
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(dtype=np.object_)
    else:
        return data.astype(np.object_)


def split_features(
    data: Union[pd.DataFrame, npt.NDArray],
    features: Optional[ft.ColumnsList],
    target: Optional[Union[str, int]],
) -> Tuple[npt.NDArray[np.object_], npt.NDArray[np.object_]]:
    if features is not None and len(features) < 1:
        ValueError("Features List cannot be empty")
    if (
        isinstance(data, np.ndarray)
        and features is not None
        and isinstance(features[0], str)
    ):
        ValueError("Expected list of integers as features, found strings")
    if isinstance(data, np.ndarray) and isinstance(target, str):
        ValueError("Expected integer as target, found string")

    # TODO: provide a proper fix instead of a ValuError
    if (target is None or features is None) and not (
        target is None and features is None
    ):
        raise ValueError(
            "Either both target and features should be provided or neither of them."
        )
    if isinstance(data, pd.DataFrame):
        if features is None:
            X = data.iloc[:, :-1]
        elif isinstance(features[0], str):
            X = data[features]
        else:
            X = data.iloc[:, features]

        if target is None:
            y = data.iloc[:, -1]
        elif isinstance(target, str):
            y = data[[target]]
        else:
            y = data.iloc[:, target]

        X = X.to_numpy()
        y = y.to_numpy()

    else:  # Numpy
        if features is None:
            X = data[:, :-1]
        else:
            X = data[:, np.asarray(features, dtype=np.int64)]

        if target is None:
            y = data[:, -1]
        else:
            y = data[:, np.asarray(target, dtype=np.int64)]

    return convert_to_np_obj(X), convert_to_np_obj(y)


def get_cat_mask(data: npt.NDArray) -> List[int]:
    num_cat_threshold: int = 10
    high_cardinality_threshold: int = 100
    mask: List[int] = []  # True -> cat ; False -> num
    tmp_df: pd.DataFrame = pd.DataFrame(data).infer_objects()
    for col in range(tmp_df.shape[-1]):
        if isinstance(
            tmp_df.iloc[-1, col],
            (int, float, np.int32, np.int64, np.float32, np.float64),
        ):
            if len(tmp_df.iloc[:, col].unique().tolist()) > num_cat_threshold:
                mask.append(0)
            else:
                mask.append(1)
        else:
            if len(tmp_df.iloc[:, col].unique().tolist()) > high_cardinality_threshold:
                mask.append(2)
            else: 
                mask.append(1)
    return mask # 0 - numerical, 1 - cat low cardinality, 2 - cat high cardinality


def calculate_model_score(y: npt.NDArray, y_hat: npt.NDArray, task: str) -> float:
    if task == "tabular_classification":
        return balanced_accuracy_score(y.astype(np.str_), y_hat)
    else:
        score = r2_score(y, y_hat)
        if score < 0:
            score = 0
        score = (score + 1) / 2
        return score


def tab_cv_score(
    pipeline: Pipeline, X: npt.NDArray, y: npt.NDArray, task: str, n_folds: int = 5
) -> List[float]:
    if task == "tabular_classification":
        kf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=1)
        y = y.astype(np.str_)
    else:
        kf = RepeatedKFold(n_splits=n_folds, n_repeats=1)
    scores = []
    for train_index, test_index in kf.split(X, y):
        copied_pipeline = deepcopy(pipeline)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        copied_pipeline.fit(X_train, y_train)
        pred = copied_pipeline.predict(X_test)
        scores.append(calculate_model_score(y_test, pred, task))
    return scores
