
# Make sure you have upgraded version of pip
Windows
```
py -m pip install --upgrade pip
```

Linux/MAC OS
```
python3 -m pip install --upgrade pip
```

## Install using pip
To install SUS libery use the following command:
```
pip install -i https://test.pypi.org/simple/ sus-libery
```
## Upgrade SUS tool is up to date
Windows, Linux/MAC OS
```
pip upgrade -i https://test.pypi.org/simple/ sus-libery
```


## User Guide: 
XEoptim is a comprehensive machine learning library that provides a wide range of optimization and scientific computing tools for solving problems in fields such as machine learning, engineering, and finance. Built on top of the popular NumPy and SciPy libraries, XEoptim offers advanced optimization capabilities including linear programming, non-linear optimization, and genetic algorithms.

In addition to optimization, XEoptim also provides a variety of machine learning algorithms such as supervised and unsupervised learning, model selection, and pre-processing. It includes a variety of classification and regression algorithms, clustering techniques, dimensionality reduction methods, and model selection tools. It also has the capability of performing neural network models, which is a state of the art technique in machine learning.

XEoptim is an open-source library and welcomes contributions from the community. It is designed to be easy to use and understand, with a consistent interface for all of its models and a large number of examples available in the documentation. With XEoptim, you can easily optimize complex machine learning models, neural network models and achieve better results. It is designed for professionals and researchers who are looking for an advanced optimization and machine learning library that provides a wide range of functionality and performance.
#### Linear and non Linear regression importing:
```
from SUS.linear_model import LinearRegression as lr
```
#### Classification algorithm importing:
```
from SUS.SVM import SVC
from SUS.LogisticRegression import LogisticRegression
from SUS.KNearestNeighbour import knn
```
#### Classification algorithm importing:
```
from SUS.SVM import SVC
from SUS.LogisticRegression import LogisticRegression
from SUS.KNearestNeighbour import knn
```
#### Unsupervised algorithm importing:
```
from SUS.EM import EM
from SUS.kmeans import kernel_km
from SUS.kmeans import km_plusplus
from SUS.kmeans import kmeans
from SUS.kmeans import loss
```
#### Sampling algorithms importing:

```
from SUS.MCMC import metropolis_hasting
```

### Model evaluation and feature selection module:
Our libery has its own model evaluation materices to evaluate regression and classification reports. and also have advance feature selection algorithms like PCA and kernals. Feel free to use it and for bugs corrections  or any contact visit my github page.

### Pip References:

https://test.pypi.org/project/sus-libery/

