Getting started
==================================

**Train a powerful Machine Learning model in a single line of code with Falcon!**

Falcon is a simple and lightweight AutoML library designed for people who want to train a model on a custom dataset in an instant even without specific data-science knowledge. Simply give Falcon your dataset and specify which feature you want the ML model to predict. Falcon will do the rest!

Falcon allows the trained models to be immediately used in production by saving them in the widely used ONNX format. No need to write custom code to save complicated models to ONNX anymore!

Installation
===================

..  code-block:: bash

    pip install git+https://github.com/OKUA1/falcon

Usage
==================

Currently, Falcon supports only tabular datasets and two tasks: 'tabular_classification' and 'tabular_regression'. 

The easiest way to use the library is to use the highest level API as shown below: 

..  code-block:: python

    from falcon import AutoML

    AutoML(task = 'tabular_classification', train_data = 'titanic.csv')


This single line of code will read and prepare the dataset, scale/encode the features, encode the labels, train the model and save as ONNX file for future inference. 

Additionally, it is also possible to explicitly specify the feature and target columns (otherwise the last column will be used as target and all other as features) and test data for evaluation report.

..  code-block:: python

    from falcon import AutoML

    manager = AutoML(task = 'tabular_classification', train_data = 'titanic.csv', test_data = 'titanic_test.csv', features = ['sex', 'gender', 'class', 'age'], target = 'survived')


It is also possible to provide train/test data as a pandas dataframe or numpy array. In order to do it, simply pass the required object as an argument. This might be relevant in cases where custom data preparation is needed or data itself comes from non conventional source. 

..  code-block:: python

    from falcon import AutoML
    import pandas as pd 

    df = pd.read_csv('titanic.csv')

    manager = AutoML(task = 'tabular_classification', train_data = df, test_data = 'titanic_test.csv', features = ['sex', 'gender', 'class', 'age'], target = 'survived')


While AutoML function enables extremely fast experementation, it does not provide enough control over the training steps and might be not flexible enough for more advanced users. As an alternative, it is possible to use the relevant TaskManager class either directly or by using :code:`initialize` helper function.

..  code-block:: python

    from falcon import initialize
    import pandas as pd 

    test_df = pd.read_csv('titanic_test.csv')

    manager = initialize(task='tabular_classification', data='titanic.csv')
    manager.train(pre_eval = True)
    manager.evaluate(test_df)
    

When using :code:`initialize` function it is also possible to provide a custom configuration or even a custom pipeline. For more details please check the API reference section.