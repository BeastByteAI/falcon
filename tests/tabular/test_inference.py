from falcon import initialize
from falcon.utils import run_onnx
import numpy as np
from sklearn.metrics import r2_score 
import random
from falcon.task_configurations import get_task_configuration

def eval_saved_model(manager, is_regr=False, format="onnx", prefix = ''):
    X = manager._data[0]
    y = manager._data[1]
    pred = manager.predict(X)
    manager.save_model(format=format, filename=f"{prefix}test_model")
    if format == "onnx":
        pred_ = run_onnx(f"{prefix}test_model.{format}", X)
        if len(pred_) > 1:
            print(
                "Onnx model returned multiple predictions. Only the first one will be used for testing."
            )
        pred_ = pred_[0].squeeze()
        print(pred_.shape)
    else:
        ValueError("Non onnx format was selected. Currently only onnx is supported.")
    if not is_regr:
        eq_ = np.equal(pred, pred_)
        print(eq_)
        return False not in eq_
    else:
        ac = np.isclose(pred, pred_)
        print(ac, np.max(pred - pred_))
        assert False not in ac
        ac = len(ac[ac == False]) 
        print(ac, len(pred))
        ac = ac / len(pred) < 0.1
        me1 = r2_score(pred, y)
        me2 = r2_score(pred_, y)
        mset = 0.001
        msec = np.abs(me1 - me2) < mset
        print(me1, me2, np.abs(me1 - me2), msec)
        print(ac)
        return ac, msec, (pred, pred_)


def inference_classification(config, config_name):
    random.seed(42)
    np.random.seed(42)
    manager = initialize(
        task="tabular_classification", data="tests/extra_files/iris.csv", **config
    )
    manager.train(pre_eval=False)
    print('model ', manager._pipeline._pipeline[1].model)
    print('task ', manager._pipeline._pipeline[1].task)
    assert eval_saved_model(manager=manager, is_regr=False, format="onnx", prefix = f"clf_{config_name}_")
    

def inference_regression(config, config_name):
    manager = initialize(
    task="tabular_regression",
    data="tests/extra_files/prices.csv",
    features="SqFt,Bedrooms,Bathrooms,Offers,Brick,Neighborhood".split(","),
    target="Price",
    )
    manager.train(pre_eval=False, **config)
    ac, msec, data =  eval_saved_model(manager=manager, is_regr=True, format="onnx", prefix = f"regr_{config_name}_")
    assert ac
    assert msec 

def test_inference_regr_superlearner_mini():
    config = get_task_configuration(task = 'tabular_regression', configuration_name='SuperLearner.mini')
    config['extra_pipeline_options']['learner_kwargs']['cv'] = 2
    inference_regression(config=config, config_name='SuperLearner.mini')

def test_inference_regr_superlearner_default():
    config = get_task_configuration(task = 'tabular_regression', configuration_name='SuperLearner')
    inference_regression(config=config, config_name='SuperLearner')

def test_inference_regr_superlearner_mid():
    config = get_task_configuration(task = 'tabular_regression', configuration_name='SuperLearner.mid')
    config['extra_pipeline_options']['learner_kwargs']['cv'] = 2
    inference_regression(config=config, config_name='SuperLearner.mid')

def test_inference_regr_superlearner_large():
    config = get_task_configuration(task = 'tabular_regression', configuration_name='SuperLearner.large')
    config['extra_pipeline_options']['learner_kwargs']['cv'] = 2
    inference_regression(config=config, config_name='SuperLearner.large')

def test_inference_regr_superlearner_xlarge():
    config = get_task_configuration(task = 'tabular_regression', configuration_name='SuperLearner.xlarge')
    config['extra_pipeline_options']['learner_kwargs']['cv'] = 2
    inference_regression(config=config, config_name='SuperLearner.xlarge')

def test_inference_clf_superlearner_mini():
    config = get_task_configuration(task = 'tabular_classification', configuration_name='SuperLearner.mini')
    config['extra_pipeline_options']['learner_kwargs']['cv'] = 2
    inference_classification(config=config, config_name='SuperLearner.mini')

def test_inference_clf_superlearner_mid():
    config = get_task_configuration(task = 'tabular_classification', configuration_name='SuperLearner.mid')
    config['extra_pipeline_options']['learner_kwargs']['cv'] = 2
    inference_classification(config=config, config_name='SuperLearner.mid')

def test_inference_clf_superlearner_large():
    config = get_task_configuration(task = 'tabular_classification', configuration_name='SuperLearner.large')
    config['extra_pipeline_options']['learner_kwargs']['cv'] = 2
    inference_classification(config=config, config_name='SuperLearner.large')

def test_inference_clf_superlearner_xlarge():
    config = get_task_configuration(task = 'tabular_classification', configuration_name='SuperLearner.xlarge')
    config['extra_pipeline_options']['learner_kwargs']['cv'] = 2
    inference_classification(config=config, config_name='SuperLearner.xlarge')

def test_inference_clf_superlearner_default():
    config = get_task_configuration(task = 'tabular_classification', configuration_name='SuperLearner')
    inference_classification(config=config, config_name='SuperLearner')

def test_inference_clf_optuna_hgbt():
    config = get_task_configuration(task = 'tabular_classification', configuration_name='OptunaLearner.hgbt')
    config['extra_pipeline_options']['learner_kwargs']['n_trials'] = 2
    inference_classification(config=config, config_name='OptunaLearnerHGBT')

def test_inference_regr_optuna_hgbt():
    config = get_task_configuration(task = 'tabular_regression', configuration_name='OptunaLearner.hgbt')
    config['extra_pipeline_options']['learner_kwargs']['n_trials'] = 2
    inference_regression(config=config, config_name='OptunaLearnerHGBT')

def test_inference_clf_plain():
    config = get_task_configuration(task = 'tabular_classification', configuration_name='PlainLearner')
    inference_classification(config=config, config_name='PlainLearner')

def test_inference_regr_plain():
    config = get_task_configuration(task = 'tabular_regression', configuration_name='PlainLearner')
    inference_regression(config=config, config_name='PlainLearner')

def test_inference_clf_plain_hgbt():
    config = get_task_configuration(task = 'tabular_classification', configuration_name='PlainLearner.hgbt')
    inference_classification(config=config, config_name='PlainLearnerHGBT')

def test_inference_regr_plain_hgbt():
    config = get_task_configuration(task = 'tabular_regression', configuration_name='PlainLearner.hgbt')
    inference_regression(config=config, config_name='PlainLearnerHGBT')

