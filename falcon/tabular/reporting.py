from numpy import typing as npt
from typing import Dict
import numpy as np
from sklearn import metrics
from falcon.tabular.utils import calculate_model_score

def print_classification_report(y: npt.NDArray, y_hat: npt.NDArray, silent: bool = False) -> None:
    y = y.astype(np.str_)
    classification_report = metrics.classification_report(y, y_hat, output_dict=True)
    metrics_ = {
        
        'ACC': metrics.accuracy_score(y, y_hat),
        'BACC': metrics.balanced_accuracy_score(y, y_hat),
        'PRECISION': list(classification_report[list(classification_report.keys())[-2]].values())[0], 
        'RECALL': list(classification_report[list(classification_report.keys())[-2]].values())[1], 
        'F1': list(classification_report[list(classification_report.keys())[-2]].values())[2], 
        'B_PRECISION': list(classification_report[list(classification_report.keys())[-1]].values())[0],
        'B_RECALL': list(classification_report[list(classification_report.keys())[-1]].values())[1], 
        'B_F1': list(classification_report[list(classification_report.keys())[-1]].values())[2]
    }
    
    metrics_['SCORE'] = metrics_['BACC']
    
    if not silent: 
        confusion_matrix = metrics.confusion_matrix(y, y_hat)
        print()
        print("PERFORMANCE REPORT: CLASSIFICATION")
        print()
        print()
        labels = [
            "Precision (% of correct predictions for this class):",
            "Recall (% of samples of this class that are correctly predicted):",
            "F1 Score (out of samples of this class, % of correct predictions):",
            "Support (number of class samples):",
        ]
        print("CLASS-SPECIFIC METRICS")
        print()
        for k in list(classification_report.keys())[:-3]:
            print(f"Label: {k}")
            print()
            for i, kk in enumerate(classification_report[k].keys()):
                print(labels[i])
                print(classification_report[k][kk])
            print()
        print()
        print("AVERAGE METRICS")
        print()
        print("Confusion Matrix")
        print("Note: high values in diagonal, low values elsewhere indicate good performance")
        print(confusion_matrix)
        print()
        print("Accuracy")
        print(
            "Note: misleading for imbalanced datasets! Low accuracy on classes with a low number of samples is not reflected!"
        )
        print(metrics_['ACC'])
        print()
        print("Balanced Accuracy")
        print("Each class weighs the same even if it has a low number of samples")
        print(metrics_['BACC'])
        print()
        # TODO 
        # print("Weighted ROC AUC score")
        # print(
        #     "Area Under the Receiver Operating Characteristic Curve, balanced by number of samples per class. The closer to 1 the better the performance."
        # )
        # print(metrics.roc_auc_score(np.eye(np.max(y) + 1)[y], np.eye(np.max(y_hat) + 1)[y_hat], multi_class="ovo"))
        # print()
        print("Precision")
        print("Note: misleading for imbalanced datasets!")
        print(metrics_['PRECISION'])
        print()
        print("Recall")
        print("Note: misleading for imbalanced datasets!")
        print(metrics_['RECALL'])
        print()
        print("F1 Score")
        print("Note: misleading for imbalanced datasets!")
        print(metrics_['F1'])
        print()
        print("Balanced Precision")
        print(metrics_['B_PRECISION'])
        print()
        print("Balanced Recall")
        print(metrics_['B_RECALL'])
        print()
        print("Balanced F1 Score")
        print(metrics_['B_F1'])
    return metrics_


def print_regression_report(y: npt.NDArray, y_hat: npt.NDArray, silent = False) -> Dict:
    metrics_ = {
        'R2': metrics.r2_score(y, y_hat), 
        'RMSE': np.sqrt(np.mean((y - y_hat) ** 2)), 
        'MSE': np.mean((y - y_hat) ** 2), 
        'MAE': np.mean(np.abs(y - y_hat)),
        'RMSLE': np.log(np.sqrt(np.mean((y - y_hat) ** 2)) + 1e-7)
    }
    metrics_['SCORE'] = calculate_model_score(y, y_hat, 'tabular_regression')
    if not silent:
        print("PERFORMANCE REPORT: REGRESSION")
        print()
        print("R2")
        print(
            "The closer to 1 the better the performance. R2 of 0 is the score of a regressor that always predicts the average."
        )
        print(metrics_['R2'])
        print()
        print("RMSE (Root Mean Squared Error)")
        print(metrics_['RMSE'])
        print()
        print("MSE (Mean Sqaured Error)")
        print(metrics_['MSE'])
        print()
        print("MAE (Mean Absolute Error)")
        print(metrics_['MAE'])
        print()
        print("RMSLE (Root Mean Squared Log Error)")
        print("Useful in case of skewed distribution of target values.")
        print(metrics_['RMSLE'])
    return metrics_
