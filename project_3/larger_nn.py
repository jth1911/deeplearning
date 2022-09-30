import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pd.read_csv("housing.data", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

# define base model
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# evaluate model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(model=larger_model, epochs=50, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10)

results = cross_val_score(pipeline, X, Y, cv=kfold, scoring='neg_mean_squared_error')

print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))