from falcon.abstract import TaskManager
from falcon.tabular import TabularTaskManager
from typing import Any, Optional, Dict, Type, Union
from falcon.abstract import Pipeline, TaskManager
import warnings
import datetime
from falcon.task_configurations import get_task_configuration

def warn(*args: Any, **kwargs: Any) -> None:
    pass


def initialize(
    task: str,
    data: Any,
    pipeline: Optional[Type[Pipeline]] = None,
    pipeline_options: Optional[Dict] = None,
    extra_pipeline_options: Optional[Dict] = None,
    features: Any = None,
    target: Any = None,
    **options: Any
) -> TaskManager:
    """
    Initializes and returns a task manager object for a given task.

    Parameters
    ----------
    task : str
        type of the task
    data : Any
        data to be used for training
    pipeline : Optional[Type[Pipeline]], optional
        class to be used as pipeline, by default None
    pipeline_options : Optional[Dict], optional
        arguments to be passed to the pipeline, by default None.
        These options will overwrite the ones from `default_pipeline_options` attribute
    extra_pipeline_options : Optional[Dict], optional
        arguments to be passed to the pipeline, by default None.
        These options will be passed in addition to the ones from `default_pipeline_options` attribute.
        This argument is ignored if `pipeline_options` is not None
    features : Any, optional
        features to be used for training, by default None
    target : Any, optional
        target to be used for training, by default None

    Returns
    -------
    TaskManager
        Initialized task manager object
    """
    warnings.warn = warn
    if task == "tabular_classification" or task == "tabular_regression":
        manager: TabularTaskManager = TabularTaskManager(
            task=task,
            data=data,
            pipeline=pipeline,
            pipeline_options=pipeline_options,
            extra_pipeline_options=extra_pipeline_options,
            features=features,
            target=target,
            **options
        )
    else:
        raise ValueError("Invalid task")

    return manager


# TODO Allow NONE for the task
def AutoML(
    task: str,
    train_data: Any,
    test_data: Any =  None,
    features: Any = None,
    target: Any = None,
    manager_configuration: Optional[Union[Dict, str]] = None
) -> TaskManager:
    """
    High level API for one line model training and evaluation.
    
    When calling the following steps will be executed:
        1) task manager object will be initialized;
        2) the model will be trained;
        3) performance summary table is printed (if test set is not provided, random split is done);
        4) the model is saved as an onnx file.

    Parameters
    ----------
    task : str
        type of the task, currently supported tasks are [`tabular_classification`, `tabular_regression`]
    train_data : Any
        data to be used for training, for tabular classification and regression this can be: path to .csv or .parquet file, pandas dataframe, numpy array, tuple (X,y)
    test_data : Any, optional
        data to be used for evaluation, for tabular classification and regression this can be: path to .csv or .parquet file, pandas dataframe, numpy array, tuple (X,y)
    features : Any, optional
        features to be used for training, for tabular classification and regression this can be: list of column names or indexes, by default None
    target : Any, optional
        target to be used for training, for tabular classification and regression this can be: column name or index, by default None
    manager_configuration : Unionp[Dict, str], optional
        task manager configuration to be used (can be used to replace pipeline/learner and/or their arguments), by default None

    Returns
    -------
    TaskManager
        Task Manager object for the corresponding task.
    """
    task = task.lower()
    if manager_configuration is None:
        manager_configuration_ = {}
    elif isinstance(manager_configuration, str): 
        manager_configuration_ = get_task_configuration(task=task, configuration_name=manager_configuration)
    manager = initialize(task=task, data=train_data, features=features, target=target, **manager_configuration_)
    make_eval_subset = True if test_data is None else False
    manager.train(pre_eval = False, make_eval_subset = make_eval_subset)
    manager.performance_summary(test_data = test_data)
    print('Saving the model ...')
    ts = datetime.datetime.now().strftime('%Y%m%d.%H%M%S')
    filename = f"falcon_{ts}.onnx"
    manager.save_model(format = 'onnx', filename = filename)
    print(f"The model was saved as `{filename}`")
    return manager
