Getting started
==================================

**Train a powerful Machine Learning model in a single line of code with Falcon!**

Falcon is a simple and lightweight AutoML library designed for people who want to train a model on a custom dataset in an instant even without specific data-science knowledge. Simply give Falcon your dataset and specify which feature you want the ML model to predict. Falcon will do the rest!

Falcon allows the trained models to be immediately used in production by saving them in the widely used ONNX format. No need to write custom code to save complicated models to ONNX anymore!

Installation
===================

Stable release from `PyPi <https://pypi.org/project/falcon-ml/>`_

..  code-block:: bash

    pip install falcon-ml

Latest version from `GitHub <https://github.com/OKUA1/falcon>`_

..  code-block:: bash

    pip install git+https://github.com/OKUA1/falcon

Installing some of the dependencies on **Apple Silicon Macs** might not work, the workaround is to create an X86 environment using `Conda <https://docs.conda.io/en/latest/>`_

..  code-block:: bash

    conda create -n falcon_env
    conda activate falcon_env
    conda config --env --set subdir osx-64
    conda install python=3.9
    pip3 install falcon-ml

Usage
==================

Currently, Falcon supports only tabular datasets and two tasks: 'tabular_classification' and 'tabular_regression'. 

The easiest way to use the library is by using the highest level API as shown below: 

..  code-block:: python

    from falcon import AutoML

    AutoML(task = 'tabular_classification', train_data = 'titanic.csv')
    

This single line of code will read and prepare the dataset, scale/encode the features, encode the labels, train the model and save it as ONNX file for future inference. 

Additionally, it is also possible to explicitly specify the feature/target columns (otherwise the last column will be used as target and all other as features) and test data (otherwise 25% of training set will be kept) for evaluation report.

..  code-block:: python

    from falcon import AutoML

    manager = AutoML(task = 'tabular_classification', train_data = 'titanic.csv', test_data = 'titanic_test.csv', features = ['sex', 'gender', 'class', 'age'], target = 'survived')


It is also possible to provide train/test data as a pandas dataframe, numpy array, or tuple containing X and y. In order to do that, simply pass the required object as an argument. This might be relevant in cases when custom data preparation is needed or data itself comes from non-conventional source. 

..  code-block:: python

    from falcon import AutoML
    import pandas as pd 

    df = pd.read_csv('titanic.csv')
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')

    manager = AutoML(task = 'tabular_classification', train_data = df, test_data = (X_test, y_test), features = ['sex', 'gender', 'class', 'age'], target = 'survived')


While AutoML function enables extremely fast experimentation, it does not provide enough control over the training steps and might be not flexible enough for more advanced users. As an alternative, it is possible to use the relevant TaskManager class either directly or by using :code:`initialize` helper function.

..  code-block:: python

    from falcon import initialize
    import pandas as pd 

    test_df = pd.read_csv('titanic_test.csv')

    manager = initialize(task='tabular_classification', data='titanic.csv')
    manager.train(make_eval_subset = True)
    manager.performance_summary(test_df)
    

When using :code:`initialize` function it is also possible to provide a custom configuration or even a custom pipeline. For more details please check the API reference section.

Demo datasets
==================

You can try out falcon using one of the built-in demo datasets. 

..  code-block:: python

    from falcon import AutoML
    from falcon.datasets import load_churn_dataset, load_insurance_dataset # churn -> classification; insurance -> regression

    df = load_churn_dataset()

    AutoML(task = 'tabular_classification', train_data = df)

Making predictions with trained models
======================

There are 2 ways to make a prediction using a trained data. If the input data is available right away, the same manager object that was used for training the model can be used. 
An important thing to notice is that the input data should have the same structure as the training set (the same number, order and type of the features). This is assumed by the model, but not explicitly checked during runtime.
The reccomended approach is to provide the data as a numpy array. 

..  code-block:: python

    from falcon import AutoML
    import pandas as pd

    df = pd.read_csv('training_data.csv')
    manager = AutoML(task = 'tabular_classification', train_data = df)

    unlabeled_data = pd.read_csv('unlabeled_data.csv').to_numpy()
    predictions = manager.predict(unlabeled_data) 
    print(predictions)

While this solution is straight-forward, in real-world applications the new/unlabaled data is not always available right away. So we want to train the model and reuse it in the future. 
TODO