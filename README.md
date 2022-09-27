# deeplearning

## Install the following packages
```
pip install scipy==1.7.3
pip install numpy==1.21.6
pip install matplotlib==3.5.3
pip install pandas==1.3.5
pip install statsmodels==0.13.2
pip install sklearn
pip install tensorflow==2.9.2
```

## Verify versions
```
# check library version numbers
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
```

## Building Deep Learning Models with Keras
With TensorFlow, you can build a neural network by creating variables for the weights and define how they interact with the training data as input to produce outputs. You already saw how to use TensorFlow this way. However, this is too tedious and inefficient. Using Keras helps you to think in terms of layers instead of individual weights and parameters.

Keras centers on the concept of a model. The main model type is a called a `Sequential` which is a linear stack of layers. You create a `Sequential` and add layers to it in the order that the computation should be performed. Once defined, you compile the model to make
 use of the underlying framework to optimize computations to be performed. In this you can
specify the loss function and the optimizer to be used.

Once compiled, the model must be fit to data. This can be done one batch of data at a time or by firing off the entire model training regime. This is where all the compute happens. Once trained, you can use your model to make predictions on new data. We can summarize the construction of deep learning models in Keras as follows:

1. __Define your model.__ Create a Sequential model and add configured layers.
2. __Compile your model.__ Specify loss function and optimizers and call the `compile()` function on the model.
3. __Fit your model.__ Train the model on a sample of data by calling the `fit()` function on the model.
4. __Make predictions.__ Use the model to generate predictions on new data by calling functions such as `evaluate()` or `predict()` on the model.
