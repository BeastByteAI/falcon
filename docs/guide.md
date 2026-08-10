# Falcon

Falcon trains tabular models and exports them to a self-contained [FNNX](https://github.com/BeastByteAI/FNNX) bundle. The bundle carries the full pipeline, so inference does not depend on the training environment.

## Installation

```bash
pip install falcon-ml
```

The base install covers training and export with scikit-learn estimators. Three optional extras add the rest:

```bash
pip install "falcon-ml[runtime]"   # load and run exported .fnnx models
pip install "falcon-ml[gbdt]"      # LightGBM, XGBoost and CatBoost candidates
pip install "falcon-ml[hpo]"       # Optuna-based hyperparameter search
```

[Compiling a model to C](#compiling-to-c) additionally needs the FNNX ahead-of-time compiler, which is not a Falcon extra:

```bash
pip install "fnnx[compiler]"
```

Falcon requires Python 3.10 or newer.

## Training a model

`AutoML` reads a dataset, infers the column types, trains a model and writes an `.fnnx` file:

```python
from falcon import AutoML

predictor = AutoML(task="tabular_classification", train_data="titanic.csv")
```

The task is either `tabular_classification` or `tabular_regression`. The training data can be a path to a `.csv` or `.parquet` file, a pandas `DataFrame`, a numpy array, or an `(X, y)` tuple. Without explicit `features` and `target`, the last column becomes the target and the rest become features.

```python
predictor = AutoML(
    task="tabular_classification",
    train_data=df,
    test_data=(X_test, y_test),
    features=["sex", "class", "age"],
    target="survived",
)
```

Passing `test_data` changes how the training data is used. With a test set, every training row is used for fitting and the test set is scored. Without one, the score comes from the training data itself, by cross-validation or by a holdout depending on the size of the dataset. [Evaluation and grouped data](#evaluation-and-grouped-data) covers that choice and how to override it.

The saved file is named `falcon_<task>_<timestamp>.fnnx`, and `save_model=False` skips writing it.

## What a run does

Every entry point runs the same steps. Falcon reads the data and infers a type for each feature column, then builds a preprocessing pipeline from those types. It trains a list of candidates one after another. A candidate is one estimator with one fixed set of hyperparameters, such as a random forest with 300 trees.

Each candidate is scored on rows it did not see while fitting, using out-of-fold predictions. Falcon then builds a weighted ensemble from the candidates, adding one at a time and keeping an addition only while the score improves.

Turning ensembling off changes only what happens after scoring. Every candidate is still trained and scored. Instead of a blend, the best-scoring one is refit on all the training rows and becomes the model. Either way, the model, the preprocessing and the label decoding are exported together as one file.

*Note: an out-of-fold prediction for a row comes from a copy of the model trained without that row. Falcon splits the data into `oof_folds` parts and rotates which part is held out, which yields one such prediction per row. Scores and ensemble weights computed this way are comparable across candidates without setting a validation set aside.*

## How the data is read

Feature columns are typed by inspection, and the type decides both the encoding and how a missing value is filled:

| Type | Detected when | Encoded as | Missing values filled with |
|---|---|---|---|
| numeric | numeric, more than ten distinct values | scaled | training median, plus an indicator column |
| categorical | numeric with ten or fewer distinct values, or any unmatched non-numeric column | one-hot up to 100 distinct values, target encoding above | sentinel category |
| text | non-numeric, detected as free text | TF-IDF, reduced with SVD | sentinel category |
| date / datetime | matches an ISO 8601 date or datetime pattern | split into components | reference value, plus a marker column |

Non-numeric columns are matched against the date and datetime patterns first, then checked for free text, and otherwise fall through to categorical.

*Note: inference runs on the data you pass, and it cannot tell an identifier from a measurement. A numeric ID with few distinct values is read as a category, and one with many distinct values is target-encoded. Drop such columns from `features` if that is not what you want.*

*Note: one-hot encoding gives every category a column of its own, which stops being practical at hundreds of categories. Target encoding instead replaces the category with a number derived from the target values seen for it in training, so one column stays one column. Falcon cross-fits that number, so a row never contributes to its own encoded value.*

Rows with a missing target are dropped before training and the count is logged. Missing feature values are kept and filled inside the pipeline. The filling is part of the exported graph, so a filled value is the same whether it comes from `Predictor.predict` or from the runtime.

## Controlling the run

`AutoML` is a wrapper around `Predictor`, which exposes the same run as separate steps:

```python
from falcon import Predictor

predictor = Predictor("tabular_classification", preset="best", time_limit=600)
predictor.fit(df, features=["sex", "class", "age"], target="survived")

predictions = predictor.predict(unlabeled_df)
probabilities = predictor.predict_proba(unlabeled_df)
metrics = predictor.evaluate(test_df)
predictor.save("model.fnnx")
```

`predict_proba` is available for classification only. `save()` returns the serialized bundle as `bytes`, and writes it to disk when given a path.

Two methods report on the run. `leaderboard()` returns a `DataFrame` with one row per trained candidate, holding its family, score, fit time and weight in the ensemble. `feature_importance(n_repeats=10)` shuffles one column at a time and measures how much the score drops, which tells you how much the model relies on that column. It scores with balanced accuracy for classification and R² for regression, and returns the features ordered by importance.

## Presets and RunConfig

Three presets ship with the library:

| Preset | Candidates | Ensembling | `oof_folds` | Plateau stop |
|---|---|---|---|---|
| `fast` | 1 | off | 2 | off |
| `balanced` (default) | up to 4 | on, up to 50 additions | 5 | on, patience 2 |
| `best` | full portfolio | on, up to 100 additions | 10 | off |

```python
predictor = Predictor("tabular_regression", preset="fast")
```

Every setting a preset controls is a field on `RunConfig`. Passing a `RunConfig` overrides only the fields you set explicitly, and the rest keep their preset values.

```python
from falcon import Predictor, RunConfig

config = RunConfig(oof_folds=10, calibrate=True, time_limit=1800)
predictor = Predictor("tabular_classification", preset="best", config=config)
```

| Field | Default | Meaning |
|---|---|---|
| `candidate_sources` | `(PortfolioSource(),)` | where candidates come from |
| `ensemble_enabled` | `True` | blend candidates; when off, the best-scoring one is refit and kept |
| `ensemble_max_iterations` | `100` | most candidates the ensemble may add |
| `plateau_enabled` | `True` | stop adding once the score stops improving |
| `plateau_patience` | `3` | additions without improvement before stopping |
| `plateau_tolerance` | `1e-4` | improvement below this counts as none |
| `oof_folds` | `5` | parts the data is split into for out-of-fold predictions |
| `eval_strategy` | `"auto"` | how the reported score is computed |
| `time_limit` | `None` | budget in seconds for the whole run |
| `random_state` | `42` | seed for splits, candidates and search |
| `dataset_aware_ordering` | `False` | try candidates in an order picked from dataset statistics |
| `calibrate` | `False` | correct overconfident probabilities (classification) |
| `conformal_alpha` | `None` | target miss rate for prediction intervals (regression) |
| `impute_missing` | `True` | fill missing values inside the pipeline |
| `class_weight` | `"none"` | reweight training rows by inverse class frequency (classification) |
| `decision_metric` | `"balanced_accuracy"` | metric the tuned decision rule maximises, or `None` for plain argmax |

`time_limit`, `random_state` and `eval_strategy` can also be passed directly to `AutoML` and `Predictor`, where they take precedence over both the preset and the config.

## Choosing candidate models

Falcon asks each source in `candidate_sources` for a list of candidates, then trains and ensembles the pooled result. The default source, `PortfolioSource`, hands back a built-in list covering linear models, random forests, extra trees and histogram gradient boosting, interleaved with LightGBM, XGBoost and CatBoost when those libraries are installed.

The cost of a run scales with the number of candidates whether ensembling is on or not. `max_candidates` shortens the built-in list for a cheaper run, and `specs` replaces the list outright:

```python
from falcon import Predictor, RunConfig
from falcon.config import PortfolioSource
from falcon.tabular.candidates import EstimatorSpec

source = PortfolioSource(
    specs=(
        EstimatorSpec("rf", "random_forest", {"n_estimators": 300}),
        EstimatorSpec("gbm", "lightgbm", {"num_leaves": 63}, min_rows=1000),
    )
)
predictor = Predictor("tabular_classification", config=RunConfig(candidate_sources=(source,)))
```

An `EstimatorSpec` names a family and the parameters to build it with. `min_rows`, `max_rows` and `max_features` decide whether it applies to the dataset at hand, so a spec meant for large data is skipped on a small one instead of trained.

`HPOSource` runs an Optuna study over one family and returns its best trials as candidates:

```python
from falcon.config import HPOSource, PortfolioSource, RunConfig

config = RunConfig(
    candidate_sources=(
        PortfolioSource(max_candidates=3),
        HPOSource(family="lightgbm", n_trials=40, top_n=3),
    )
)
```

Sources run in order and their candidates go into one pool, so a fixed portfolio and a search can feed the same ensemble. `time_budget_fraction` caps the share of the remaining time limit the study may spend, and defaults to a quarter.

*Note: `HPOSource` requires the `hpo` extra, and GBDT families require the `gbdt` extra. A family whose library is missing is dropped from the default portfolio with a log message, but naming it explicitly in an `EstimatorSpec` raises an `ImportError`.*

## Evaluation and grouped data

`eval_strategy` decides how the reported score is produced. Under `"auto"`, datasets below 2,500 rows are scored with cross-validation and larger ones with a 25% holdout. `"holdout"` and `"cv"` force either choice. `None` skips evaluation and fits on every row, which is what happens when you pass `test_data` to `AutoML`. A scikit-learn cross-validator or a callable returning train and test indices can be passed instead of a name.

When rows are not independent, `group_by` keeps a group on one side of every split:

```python
predictor.fit(
    df,
    features=["customer_id", "sex", "age"],
    target="churn",
    group_by="customer_id",
)
```

`group_by` accepts a column name, several column names, or an array of group labels aligned with the rows of the input data. A named column has to be one of the selected `features`. To group on a column the model should not see, pass its values as an array instead. Grouping applies to the evaluation split and to the out-of-fold folds, which the calibration and interval settings below also rely on.

*Note: without `group_by`, rows are grouped by their full feature vector. Duplicate rows therefore stay on the same side of a split, which keeps an exact copy of a training row out of the evaluation set.*

## Calibrated probabilities

A classifier can rank cases correctly and still report probabilities that are too confident. Of all the cases a model calls 90% likely, about 90% should actually turn out positive. Calibration closes that gap.

`calibrate=True` fits a temperature on the out-of-fold predictions. The temperature is a single number that divides the model's scores before they become probabilities, which makes every probability softer or sharper by the same factor. Falcon bakes it into the exported graph, so `predict_proba` returns calibrated values both natively and through the runtime. Temperature alone preserves the ordering of the scores, so on its own it moves the probabilities and not the labels. With the tuned decision rule below it can move labels too, because the rule reads the calibrated scores and a weighted comparison between three or more classes is not preserved by a temperature. Regression models reject the setting.

## Imbalanced classes

When one class is much rarer than the others, a model trained to predict the most likely class will rarely predict the rare one. Falcon addresses this at the decision, not at the probabilities.

`decision_metric` fits one weight per class on the out-of-fold predictions and picks the label at `argmax(p * w)` instead of `argmax(p)`. In a two-class problem that is exactly a tuned threshold; with more classes it is a per-class tilt. The weights go into the exported graph as two nodes, so native and runtime predictions agree. The default, `"balanced_accuracy"`, matches the score Falcon reports. `"f1"` and `"mcc"` are also accepted, and `None` turns the rule off and restores plain argmax. `"f1"` is macro-averaged on two classes as well as on more, because Falcon encodes labels alphabetically and neither class of a two-class target is inherently the positive one. Falcon leaves the weights at one when the rarest class has fewer than 50 out-of-fold rows, or when no weighting beats plain argmax on the metric.

The rule changes labels only. `predict_proba` returns the same numbers with the rule on or off, which means code that thresholds `predict_proba` at 0.5 itself bypasses the tuned rule entirely and keeps the untuned decision.

`class_weight="balanced"` is the other lever, and it works the other way around: it reweights the training rows by inverse class frequency, which shifts the probabilities themselves rather than the decision taken from them. It costs log loss and calibration quality and largely duplicates what the decision rule already does, so it stays off by default.

*Note: optimising balanced accuracy trades against plain accuracy, since predicting the rare class more often costs errors on the common one. `evaluate()` reports both.*

## Prediction intervals

A regression model returns one number per row. `conformal_alpha` adds a lower and an upper bound around it.

Falcon measures how far the out-of-fold predictions land from the true values, then takes the quantile of those distances that `alpha` implies. That distance becomes a fixed margin added on both sides of every prediction. With `alpha=0.1`, roughly 90% of future rows should fall inside their interval. The exported model then has three outputs instead of one: `y_pred`, `y_lower` and `y_upper`.

```python
from falcon import Predictor, RunConfig
from falcon.runtime import Runtime

predictor = Predictor("tabular_regression", config=RunConfig(conformal_alpha=0.1))
predictor.fit(df, target="charges")
predictor.save("model.fnnx")

lower, upper = Runtime("model.fnnx").predict_interval(X)
```

The 90% holds across rows on average, not for any single row, and only while new data resembles the data the margin was measured on. Every interval has the same width, so a row the model finds hard is not given a wider one.

## Export and inference

`save()` writes a single `.fnnx` file containing the preprocessing, the model or ensemble, and the label decoding. The `runtime` extra provides a thin wrapper for loading it:

```python
from falcon.runtime import Runtime

runtime = Runtime("model.fnnx")
predictions = runtime.predict(unlabeled_df)
probabilities = runtime.predict_proba(unlabeled_df)
```

The runtime accepts a `DataFrame`, a numpy array, or a dict of column arrays. Columns have to arrive in the same order and with the same types as during training. This is assumed rather than checked, so a reordered frame produces wrong numbers instead of an error. Classification models return decoded labels, matching what `Predictor.predict` returns.

## Models without inference-time imputation

Not every deployment target accepts a graph whose data path depends on the values flowing through it. Filling a missing value at inference time creates such a dependency, because it means choosing between two values per row. In ONNX that is a `Where` node, fed by an `IsNaN` test for numeric columns and by `Equal`/`Or` comparisons against four missing tokens for string columns. The data path through the graph then depends on the values flowing through it.

`impute_missing=False` builds the pipeline without that handling:

```python
from falcon import AutoML, RunConfig

AutoML(task="tabular_classification", train_data=df, config=RunConfig(impute_missing=False))
```

Numeric columns are then cast to `float32` and nothing else, which drops both the choice and the indicator column. There is no way to fill a missing number without that choice, so Falcon refuses at fit time and names the column. Categorical and text columns keep their plain string form, which turns a missing value into an ordinary category (`"nan"`, `"None"`, and so on) instead of a shared sentinel.

Date and datetime features are rejected in this mode. Their tokenizer fills missing values internally and needs the same choice to do it, so Falcon fails with an error listing the offending columns rather than quietly emitting one.

The resulting graph holds no `Where`, `IsNaN`, `Equal`, `Or`, `If`, `Loop` or `Scan` nodes for numeric, categorical and text features. In exchange, missing values at inference no longer have defined behavior. A missing number travels through the model as `NaN`, and a missing category is treated as one the model never saw.

## Compiling to C

An exported model can also be compiled to C source instead of being loaded through the runtime. The generated code carries the whole pipeline, from the scaling and encoding through to every tree of the ensemble, and needs no runtime, no allocation and no ONNX at inference time. `compile_to_c` reads a `.fnnx` file and writes the C into a directory you name:

```python
from falcon.codegen import compile_to_c

compile_to_c("model.fnnx", "out/", prefix="charges", batch_size=32)
```

Three files land in the output directory. `charges.h` holds the model as straight-line C, `charges_falcon.h` holds the string tables the next section covers, and `charges_report.json` describes the artifact for tooling. Both headers follow the single-header convention: include them anywhere, and define the implementation macro in exactly one translation unit.

```c
#define CHARGES_IMPLEMENTATION
#include "charges.h"

#define CHARGES_FALCON_IMPLEMENTATION
#include "charges_falcon.h"

cat_color[0] = charges_encode_cat_color("red");
charges_run(rows, num_a, num_b, cat_color, cat_size, y_pred);
```

`batch_size` is the largest number of rows one call may pass, and it fixes the size of the buffers the artifact reserves. A call can always pass fewer rows.

Generating C requires FNNX's ahead-of-time compiler, the `fnnx.extras.compilers.c` module that `pip install "fnnx[compiler]"` brings in. Without it the call raises `CodegenError` saying as much.

## Categories and labels in compiled C

C has no string tensor, so the two places Falcon puts one have to be resolved before the graph is compiled. Categorical features arrive as int64 category codes rather than strings, and a classifier returns the predicted class as an int64 index rather than a label. This happens during code generation, not during export: the `.fnnx` file is not modified and keeps its string interface for every other consumer.

The mapping is not lost. It moves into `<prefix>_falcon.h`. Each categorical feature gets its vocabulary as a table and a lookup that returns the code for a string, and a classifier gets its class labels and a lookup that returns the label for an index. The header also documents the argument order of `<prefix>_run()` and what each argument holds.

```c
int64_t charges_encode_cat_color(const char* value);
const char* charges_class_label(int64_t index);
```

A value in no category encodes to `-1`, which the model treats the way it treats any category it did not see during training: as an all-zero encoding, not as an error.

Imputation needs no special handling. When the model was trained with `impute_missing=True`, the generated lookup maps `NULL` and the missing tokens Falcon recognizes to the sentinel category the pipeline fills with, so a missing value reaches the model as the same category it would have in Python. Numeric imputation compiles as it stands, `IsNaN` and `Where` included.

*Note: text and date features cannot be compiled. Both are encoded by splitting and parsing the string itself rather than by looking the whole value up, so no integer code stands in for one. `compile_to_c` raises `CodegenError` naming the offending columns. Drop them from `features`, or deploy that model through the FNNX runtime instead.*

## The scikit-learn API

`FalconTabularClassifier` and `FalconTabularRegressor` wrap `Predictor` behind the estimator interface, which lets Falcon sit inside scikit-learn tooling that expects `fit`/`predict`:

```python
from falcon.sklapi import FalconTabularClassifier

model = FalconTabularClassifier(preset="balanced")
model.fit(X_train, y_train, group_by="customer_id")
model.predict(X_test)
model.save_model("model.fnnx")
```

`preset` takes either a preset name or a `RunConfig`. `FalconClassifier` and `FalconRegressor` are aliases for the same classes.

## Demo datasets

Two datasets are bundled for trying the library out. `load_churn_dataset` suits classification and `load_insurance_dataset` suits regression. Both take `mode="training"` for a labelled `DataFrame` and `mode="inference"` for an unlabelled array.

```python
from falcon import AutoML
from falcon.datasets import load_churn_dataset

AutoML(task="tabular_classification", train_data=load_churn_dataset())
```
