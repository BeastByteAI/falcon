from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler


class BalancedStratifiedKFold(StratifiedKFold):
    def split(self, X, y, groups=None): # type: ignore
        for train, test in super().split(X, y, groups):
            y_train_split = y[train]
            y_test_split = y[test]
            train, _ = RandomOverSampler().fit_resample(
                train.reshape(-1, 1), y_train_split
            )

            
            yield train.squeeze(), test.squeeze()
