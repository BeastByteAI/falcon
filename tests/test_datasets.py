from falcon.datasets import load_churn_dataset, load_insurance_dataset

def test_churn_dataset():
    df = load_churn_dataset()
    assert df is not None 
    assert df.shape == (10000, 11)


def test_insurance_dataset():
    df = load_insurance_dataset()
    assert df is not None 
    assert df.shape == (1338, 7)