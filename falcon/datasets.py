import pandas as pd
from typing import Union
import numpy as np
from falcon.utils import print_


def load_churn_dataset(mode: str = "training") -> Union[pd.DataFrame, np.ndarray]:
    print_("Loading churn dataset ...")
    df = pd.read_csv(
        "https://gist.githubusercontent.com/OKUA1/b5faf7b5b3fa9d69bbb64b52670ecf10/raw/d5f87274ad244f3da4b9e330bf7fc9a8d3015f0b/churn.csv"
    )
    if mode == "training":
        print_(df.head(5))
        print_(f"Dataset shape: {df.shape}")
        print_("This dataset can be used for `tabular_classification` task")
    elif mode == "inference":
        df.pop("churn")
        df = df.to_numpy()
    else:
        raise ValueError(f"Unknown mode {mode}, expected `training` or `inference`")
    return df


def load_insurance_dataset(mode: str = "training") -> Union[pd.DataFrame, np.ndarray]:
    print_("Loading insurance dataset ...")
    df = pd.read_csv(
        "https://gist.githubusercontent.com/OKUA1/b5faf7b5b3fa9d69bbb64b52670ecf10/raw/d5f87274ad244f3da4b9e330bf7fc9a8d3015f0b/insurance.csv"
    )
    if mode == "training":
        print_(df.head(5))
        print_(f"Dataset shape: {df.shape}")
        print_("This dataset can be used for `tabular_regression` task")
    elif mode == "inference":
        df.pop("charges")
        df = df.to_numpy()
        print(df)
    else:
        raise ValueError(f"Unknown mode {mode}, expected `training` or `inference`")
    return df

