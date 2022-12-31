Available Configurations
==============================

.. list-table:: configurations for tabular_regression/tabular_classification tasks
   :widths: 25 75
   :header-rows: 1

   * - Name
     - Description
   * - SuperLearner
     - | Uses :doc:`tabular/learners/super_learner` to build a stacking ensemble of base estimators. 
       | The sub-config is determined by the dataset size.
   * - SuperLearner.mini
     - | Uses :doc:`tabular/learners/super_learner` with a config for small datasets. 
       | The dataset is considered small when the number of cells after preprocessing [n_rows*n_columns] is < 80k.
   * - SuperLearner.mid
     - | Uses :doc:`tabular/learners/super_learner` with a config for mid datasets. 
       | The dataset is considered small when the number of cells after preprocessing [n_rows*n_columns] is < 4kk.
   * - SuperLearner.large
     - | Uses :doc:`tabular/learners/super_learner` with a config for large datasets. 
       | The dataset is considered small when the number of cells after preprocessing [n_rows*n_columns] is < 16kk.
   * - SuperLearner.xlarge
     - | Uses :doc:`tabular/learners/super_learner` with a config for x-large datasets. 
       | The dataset is considered small when the number of cells after preprocessing [n_rows*n_columns] is >= 16kk.
   * - OptunaLearner
     - | Uses :doc:`tabular/learners/optuna_learner`;
       | builds a model with hyperparameters optimized by Optuna framework and HistGradientBoosting as default model. 
   * - OptunaLearner.hgbt
     - | Alias for 'OptunaLearner';
       | HistGradientBoosting is used explicitly instead of relying on :doc:`tabular/learners/optuna_learner` default choice.
