from typing import Any

import pandas as pd
from numpy import typing as npt

from falcon.utils import logger


def load_churn_dataset(
    mode: str = "training",
) -> pd.DataFrame | npt.NDArray[Any]:
    logger.info("Loading churn dataset ...")
    df = pd.read_csv(
        "https://gist.githubusercontent.com/OKUA1/b5faf7b5b3fa9d69bbb64b52670ecf10/raw/d5f87274ad244f3da4b9e330bf7fc9a8d3015f0b/churn.csv"
    )
    if mode == "training":
        logger.info("%s", df.head(5))
        logger.info("Dataset shape: %s", df.shape)
        logger.info("This dataset can be used for `tabular_classification` task")
    elif mode == "inference":
        df.pop("churn")
        df = df.to_numpy()
    else:
        raise ValueError(f"Unknown mode {mode}, expected `training` or `inference`")
    return df


def load_insurance_dataset(
    mode: str = "training",
) -> pd.DataFrame | npt.NDArray[Any]:
    logger.info("Loading insurance dataset ...")
    df = pd.read_csv(
        "https://gist.githubusercontent.com/OKUA1/b5faf7b5b3fa9d69bbb64b52670ecf10/raw/d5f87274ad244f3da4b9e330bf7fc9a8d3015f0b/insurance.csv"
    )
    if mode == "training":
        logger.info("%s", df.head(5))
        logger.info("Dataset shape: %s", df.shape)
        logger.info("This dataset can be used for `tabular_regression` task")
    elif mode == "inference":
        df.pop("charges")
        df = df.to_numpy()
        logger.debug("Loaded inference data: %s", df)
    else:
        raise ValueError(f"Unknown mode {mode}, expected `training` or `inference`")
    return df
