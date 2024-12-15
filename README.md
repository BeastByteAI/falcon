<p align="center">
<picture>
  <img alt="Falcon logo" src="https://gist.githubusercontent.com/OKUA1/55e2fb9dd55673ec05281e0247de6202/raw/41063fcd620d9091662fc6473f9331a7651b4465/falcon.svg" height = "250">
</picture>
</p>


# FALCON: A Lightweight AutoML Library
Falcon is a lightweight python library that allows to train production-ready machine learning models in a single line of code. 

## Why Falcon ? 🔍

- Simplicity: With Falcon, training a comprehensive Machine Learning pipeline is as easy as writing a single line of code.
- Flexibility: Falcon offers a range of pre-set configurations, enabling swift interchangeability of internal components with just a minor parameter change.
- Portability: A standout feature of Falcon is its deep native support for [FNNX](https://github.com/BeastByteAI/FNNX)/[ONNX](https://onnx.ai/) models. This lets you export complex pipelines into a single production-ready file, irrespective of the underlying frameworks. As a result, your model can be conveniently deployed without any dependency on the training environment.

⭐ If you liked the project, please support us with a star!

## Quick Start 🚀

You can try falcon out simply by pointing it to the location of your dataset.

```python
from falcon import AutoML

AutoML(task = 'tabular_classification', train_data = '/path/to/titanic.csv')
```

Alternatively, you can use one of the available demo datasets.

```python
from falcon import AutoML
from falcon.datasets import load_churn_dataset, load_insurance_dataset 
# churn -> classification; insurance -> regression

df = load_churn_dataset()

AutoML(task = 'tabular_classification', train_data = df)
```

## Installation 💾 

Stable release from [PyPi](https://pypi.org/project/falcon-ml/)
```bash 
pip install falcon-ml
```

Latest version from [GitHub](https://github.com/OKUA1/falcon)
```bash
pip install git+https://github.com/OKUA1/falcon
```

## Documentation 📚
You can find a more detailed guide as well as an API reference in our [official docs](https://beastbyteai.github.io/falcon/intro.html#).

