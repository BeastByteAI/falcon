from falcon.tabular import reporting
import numpy as np


def test_classification_report(): 
    y = np.random.randint(0, 3, size=100)
    y_hat = np.random.randint(0, 3, size=100)

    metrics = reporting.print_classification_report(y, y_hat, silent = True)

    assert isinstance(metrics, dict)
    required_metrics = ['ACC', 'BACC', 'PRECISION', 'RECALL', 'F1', 'B_PRECISION', 'B_RECALL', 'B_F1', 'SCORE']
    for rm in required_metrics: 
        assert rm in metrics.keys()
        assert metrics[rm] is not None
    #assert isinstance(metrics['CONF_MAT'], list) 
    assert metrics['SCORE'] >= 0 and metrics['SCORE'] <= 1

def test_regression_report(): 
    y = np.random.normal(0, 1, size=10000)
    y_hat = np.random.normal(0, 1.25, size=10000)

    metrics = reporting.print_regression_report(y, y_hat, silent = True)

    assert isinstance(metrics, dict)
    required_metrics = ['SCORE', 'R2', 'RMSE', 'MSE', 'MAE', 'RMSLE']
    for rm in required_metrics: 
        assert rm in metrics.keys()
        assert metrics[rm] is not None
    assert metrics['SCORE'] >= 0 and metrics['SCORE'] <= 1
    assert metrics['MSE'] >= 0
    assert metrics['RMSE'] >= 0 
    assert metrics['MAE'] >= 0