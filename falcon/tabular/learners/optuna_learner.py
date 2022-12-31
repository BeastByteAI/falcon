from falcon.abstract import Learner
from typing import Type, Optional, Any, Union, Dict, Callable
from falcon.types import Float32Array, Int64Array
from numpy import typing as npt
from falcon.serialization import SerializedModelRepr
from falcon.abstract import Model, ONNXConvertible
from falcon.abstract.optuna import OptunaMixin
from falcon.tabular.models.hist_gbt import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
import optuna
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

class OptunaLearner(Learner, ONNXConvertible):
    def __init__(
        self, task: str, model_class: Optional[Type] = None, n_trials: Optional[int] = None, **kwargs: Any
    ) -> None:
        """
        OptunaLerner select the best hyperparameters for the given model using the Optuna Framework.

        Parameters
        ----------
        task : str
            'tabular_classification' or 'tabular_regression'
        model_class : Optional[Type], optional
            the class of the model to train, by default None;
            if None, HistGradientBoosting
        n_trials : Optional[int], optional
            number of optimization trials, minimum 20, by default None;
            if None, the number of trials is chosen dynamically based on the dataset size

        """
        self.task = task
        if model_class is None: 
            if task == "tabular_classification": 
                model_class = HistGradientBoostingClassifier
            elif task == "tabular_regression":
                model_class = HistGradientBoostingRegressor
            else:
                ValueError("Not supported task")
        if not issubclass(model_class, Model) or not issubclass(model_class, OptunaMixin):
            raise ValueError('Model class should be a subclass of falcon.abstract.Model')
        if not issubclass(model_class, ONNXConvertible):
            raise ValueError('OptunaLearner only supports ONNXConvertible models')
        
        if n_trials is not None and n_trials < 5:
            print('n_trials should be >= 20, setting n_trials = 20')
            n_trials = 20

        self.n_trials = n_trials
        
        self.model_class = model_class

    def _make_objective_func(self, search_space: Dict, X: npt.NDArray, y: npt.NDArray) -> Callable:
        stratify = y if self.task == "tabular_classification" else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, stratify=stratify)
        
        if self.task == 'tabular_regression':
            loss = mean_squared_error
        elif self.task == 'tabular_classification':
            loss = lambda y, y_h : - balanced_accuracy_score(y,y_h)
            X_train, y_train = RandomOverSampler().fit_resample(
                X_train, y_train
            )
        progress_bar = tqdm(total = self.n_trials)
        def objective(trial) -> float:
            params = {}
            for hp_n, hp_v in search_space.items():
                if hp_v["type"] == "int":
                    params[hp_n] = trial.suggest_int(name = hp_n, **hp_v["kwargs"])
                if hp_v["type"] == "float":
                    params[hp_n] = trial.suggest_float(name = hp_n, **hp_v["kwargs"]) 
                if hp_v["type"] == "categorical":
                    params[hp_n] = trial.suggest_categorical(name = hp_n, **hp_v["kwargs"]) 
            
            model = self.model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            loss_ = loss(y_val, y_pred)
            progress_bar.update(1)
            return loss_
        self.progress_bar = progress_bar
        return objective

    
    def _set_n_trials(self, X: npt.NDArray, y: npt.NDArray) -> None:
        if self.n_trials is not None: 
            return 
        
        volume = X.shape[0] * X.shape[1]

        min_threshold = 80_000  # 5_000 samples with 16 features
        mid_threshold = 4_000_000  # 125_000 samples with 32 featrues / 250_000 samples with 16 features
        large_threshold = 16_000_000 # 1_000_000 samples with 16 features

        if volume < min_threshold:
            self.n_trials = 500
        elif volume < mid_threshold:
            self.n_trials = 200
        elif volume < large_threshold:
            self.n_trials = 100
        else: 
            self.n_trials = 30

    def fit(self, X: Float32Array, y: Float32Array) -> None:
        """
        Fits the model by choosing the best hyperparameters and training the final model using them.
        For classification tasks, the dataset will be balanced by upsampling the minority class(es).

        Parameters
        ----------
        X : Float32Array
            features
        y : Float32Array
            targets
        """
        self._set_n_trials(X, y)
        search_space = self.model_class.get_search_space(X, y)
        self.progress_bar = None
        if isinstance(search_space, dict):
            objective = self._make_objective_func(search_space, X, y)
        else: 
            NotImplementedError('Non-dict search space is not yet supported.')
        
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials)
        if self.progress_bar:
            self.progress_bar.close()
        best_params = study.best_params

        if self.task == 'tabular_classification':
            X, y = RandomOverSampler().fit_resample(
                X, y
            )
        model = self.model_class(**best_params)
        model.fit(X,y)
        self.model = model
    
    def predict(self, X: Float32Array) -> Union[Float32Array, Int64Array]:
        return self.model.predict(X)

    def get_input_type(self) -> Type:
        """
        Returns
        -------
        Type
            Float32Array
        """
        return Float32Array

    def get_output_type(self) -> Type:
        """
        Returns
        -------
        Type
            Float32Array for regression, Int64Array for classification
        """
        return Float32Array if self.task == "tabular_regression" else Int64Array
    
    def forward(self, X: Float32Array) -> Union[Float32Array, Int64Array]:
        """
        Equivalent to `.predict(X)`

        Parameters
        ----------
        X : Float32Array
            features

        Returns
        -------
        Union[Float32Array, Int64Array]
            predictions
        """
        return self.model.predict(X)

    def fit_pipe(self, X: Float32Array, y: Float32Array) -> None:
        """
        Equivalent to `.fit(X, y)`

        Parameters
        ----------
        X : Float32Array
            features
        y : Float32Array
            targets
        """
        self.fit(X, y)

    def to_onnx(self) -> SerializedModelRepr:
        """
        Serializes the underlying model to onnx by calling its `.to_onnx()` method.

        Returns
        -------
        SerializedModelRepr
        """
        return self.model.to_onnx()
        