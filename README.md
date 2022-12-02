
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
pip install sus
```
## Upgrade SUS tool is up to date
Windows
```
py -m pip install --upgrade SUS
```
Linux/MAC OS
```
python3 -m pip install --upgrade SUS
```


## User Guide: 
SUS is a simple machine learning libery capable of predictive data analysis. Libery contains all supervised algorithms such as linear and non linear regression models and classification algorithms such as support vector machine, logistic regression, k nearest neighbour algorithms, decission trees etc. Unsupervised module of supervised module currently have the clustering algorithms in version-0.0 such as kmean, kmean++ and expecatation maximization. Last module is sampling algorithm (MCMC) and neural networks(MLP).
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

### GitHub References:

https://github.com/mlayek21/SUS-Libery.git

