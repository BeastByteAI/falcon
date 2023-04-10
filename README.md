<p align="center">
  <img src="https://raw.githubusercontent.com/OKUA1/falcon/main/docs/source/logo_cropped.png" width="256" height="217"/>
</p>


# FALCON: A Lightweight AutoML Library
Falcon is a lightweight python library that allows to train production-ready machine learning models in a single line of code. 

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

Installing some of the dependencies on **Apple Silicon Macs** might not work, the workaround is to create an X86 environment using [Conda](https://docs.conda.io/en/latest/)

```bash 
conda create -n falcon_env
conda activate falcon_env
conda config --env --set subdir osx-64
conda install python=3.9
pip3 install falcon-ml
```

## Documentation 📚
You can find a more detailed guide as well as an API reference in our [official docs](https://okua1.github.io/falcon/intro.html#).

## Authors & Contributors ✨
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://www.linkedin.com/in/oleh-kostromin-b671a4157/"><img src="https://avatars.githubusercontent.com/u/48349467?v=4" width="100px;" alt=""/><br /><sub><b>Oleg Kostromin</b></sub></a><br /></td>
      <td align="center"><a href="https://www.linkedin.com/in/iryna-kondrashchenko-673800155/"><img src="https://avatars.githubusercontent.com/u/72279145?v=4" width="100px;" alt=""/><br /><sub><b>Iryna Kondrashchenko</b></sub></a><br /></td>
      <td align="center"><a href="https://www.linkedin.com/in/pasinimarco/"><img src="https://avatars.githubusercontent.com/u/50598094?v=4" width="100px;" alt=""/><br /><sub><b>Marco Pasini</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>
