from numpy import typing as npt
import numpy as np
import sklearn
from sklearn import metrics


def print_classification_report(y: npt.NDArray, y_hat: npt.NDArray) -> None:
    y = y.astype(np.str_)
    print()
    print("PERFORMANCE REPORT: CLASSIFICATION")
    print()
    print()
    classification_report = metrics.classification_report(y, y_hat, output_dict=True)
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
    print(metrics.confusion_matrix(y, y_hat))
    print()
    print("Accuracy")
    print(
        "Note: misleading for imbalanced datasets! Low accuracy on classes with a low number of samples is not reflected!"
    )
    print(metrics.accuracy_score(y, y_hat))
    print()
    print("Balanced Accuracy")
    print("Each class weighs the same even if it has a low number of samples")
    print(metrics.balanced_accuracy_score(y, y_hat))
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
    print(list(classification_report[list(classification_report.keys())[-2]].values())[0])
    print()
    print("Recall")
    print("Note: misleading for imbalanced datasets!")
    print(list(classification_report[list(classification_report.keys())[-2]].values())[1])
    print()
    print("F1 Score")
    print("Note: misleading for imbalanced datasets!")
    print(list(classification_report[list(classification_report.keys())[-2]].values())[2])
    print()
    print("Balanced Precision")
    print(list(classification_report[list(classification_report.keys())[-1]].values())[0])
    print()
    print("Balanced Recall")
    print(list(classification_report[list(classification_report.keys())[-1]].values())[1])
    print()
    print("Balanced F1 Score")
    print(list(classification_report[list(classification_report.keys())[-1]].values())[2])
    pass


def print_regression_report(y: npt.NDArray, y_hat: npt.NDArray) -> None:
    print("PERFORMANCE REPORT: REGRESSION")
    print()
    print("R2")
    print(
        "The closer to 1 the better the performance. R2 of 0 is the score of a regressor that always predicts the average."
    )
    print(metrics.r2_score(y, y_hat))
    print()
    print("RMSE (Root Mean Squared Error)")
    print(np.sqrt(np.mean((y - y_hat) ** 2)))
    print()
    print("MSE (Mean Sqaured Error)")
    print(np.mean((y - y_hat) ** 2))
    print()
    print("MAE (Mean Absolute Error)")
    print(np.mean(np.abs(y - y_hat)))
    print()
    print("RMSLE (Root Mean Squared Log Error)")
    print("Useful in case of skewed distribution of target values.")
    print(np.log(np.sqrt(np.mean((y - y_hat) ** 2)) + 1e-7))
    pass
