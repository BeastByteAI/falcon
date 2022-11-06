import pandas as pd
from falcon.utils import print_


def load_churn_dataset() -> pd.DataFrame:
    print_("Loading churn.csv ...")
    df = pd.read_csv(
        "https://gist.githubusercontent.com/OKUA1/b5faf7b5b3fa9d69bbb64b52670ecf10/raw/d5f87274ad244f3da4b9e330bf7fc9a8d3015f0b/churn.csv"
    )
    print_(df.head(5))
    print_(f"Dataset shape: {df.shape}")
    print_("This dataset can be used for `tabular_classification` task")
    return df


def load_insurance_dataset() -> pd.DataFrame:
    print_("Loading insurance.csv ...")
    df = pd.read_csv(
        "https://gist.githubusercontent.com/OKUA1/b5faf7b5b3fa9d69bbb64b52670ecf10/raw/d5f87274ad244f3da4b9e330bf7fc9a8d3015f0b/insurance.csv"
    )
    print_(df.head(5))
    print_(f"Dataset shape: {df.shape}")
    print_("This dataset can be used for `tabular_regression` task")
    return df
