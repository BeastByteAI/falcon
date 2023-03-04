Available Configurations
==============================

The tables below list both main and additional configurations that can be used. 
Additional configurations should be used with caution as they may not be suitable for certain datasets. It is reccomended to always choose one of the main configurations.

***********************************************************************************
Configurations for tabular_regression/tabular_classification tasks
***********************************************************************************

.. list-table::
   :width: 100%
   :widths: 18 12 70
   :header-rows: 1

   * - Name
     - Extension
     - Description
   * - SuperLearner
     - --
     - | Uses :doc:`tabular/learners/super_learner` to build a stacking ensemble of base estimators. 
       | SuperLearner combines multiple individual estimators to make predictions with greater accuracy than any of the individual estimators alone. 
       | Additionaly, it learns to weigh the predictions of each individual model, optimizing the combination to maximize performance on the given task.
       | SuperLearner is more suitable for smaller datasets, but the produced models tend to be relatively large.
   * - OptunaLearner
     - --
     - | Uses :doc:`tabular/learners/optuna_learner`.
       | It builds a model and optimizes its hyperparameters using Optuna framework; :doc:`tabular/models/hgbt_clf`/:doc:`tabular/models/hgbt_regr` is used as a default model. 
       | Since OptunaLearner focuses on finetuning a single model, the produced model is not very large in size, but the optimization procedure can be very long.
   * - PlainLearner
     - --
     - | Uses :doc:`tabular/learners/plain_learner`.
       | It builds a model using default hyperparameters; :doc:`tabular/models/hgbt_clf`/:doc:`tabular/models/hgbt_regr` is used as a default model.
       | PlainLearner is very fast, thus it is a good choice for building initial baselines or automizing preprocessing steps.

.. dropdown:: Additional configurations

  .. list-table::
    :width: 100%
    :widths: 18 12 70
    :header-rows: 1

    * - Name
      - Extension
      - Description
    * - SuperLearner.mini
      - --
      - | Uses :doc:`tabular/learners/super_learner` with a config for small datasets. 
        | The dataset is considered small when the number of cells after preprocessing [n_rows*n_columns] is < 80k.
    * - SuperLearner.mid
      - --
      - | Uses :doc:`tabular/learners/super_learner` with a config for mid datasets. 
        | The dataset is considered small when the number of cells after preprocessing [n_rows*n_columns] is < 4kk.
    * - SuperLearner.large
      - --
      - | Uses :doc:`tabular/learners/super_learner` with a config for large datasets. 
        | The dataset is considered small when the number of cells after preprocessing [n_rows*n_columns] is < 16kk.
    * - SuperLearner.xlarge
      - --
      - | Uses :doc:`tabular/learners/super_learner` with a config for x-large datasets. 
        | The dataset is considered small when the number of cells after preprocessing [n_rows*n_columns] is >= 16kk.
    * - OptunaLearner.hgbt
      - --
      - | Uses :doc:`tabular/learners/optuna_learner`.
        | It builds a :doc:`tabular/models/hgbt_clf`/:doc:`tabular/models/hgbt_regr` model with hyperparameters optimized by Optuna framework. 
    * - PlainLearner.hgbt
      - --
      - | Uses :doc:`tabular/learners/plain_learner`.
        | It builds a :doc:`tabular/models/hgbt_clf`/:doc:`tabular/models/hgbt_regr` model with default hyperparameters. 

