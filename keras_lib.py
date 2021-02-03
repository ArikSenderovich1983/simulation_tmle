# scipy
import scipy
#print('scipy: %s' % scipy.__version__)
# matplotlib
#import matplotlib
#print('matplotlib: %s' % matplotlib.__version__)

# statsmodels
import statsmodels
#print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
#print('sklearn: %s' % sklearn.__version__)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# theano
#import theano
#print('theano: %s' % theano.__version__)
# tensorflow
#import tensorflow
#print('tensorflow: %s' % tensorflow.__version__)
# keras
import pandas as pd
import numpy as np
#import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation,concatenate
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.utils import plot_model

#from tensorflow import keras
from keras.layers import BatchNormalization

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Loading Data
# Data disposition
from keras import backend as K
from keras.regularizers import l2
from keras.regularizers import l1
from keras.layers import Dropout

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def approx_Lindley(df, target_col, num_cols):
    #print(df.head())
    #print(len(df))
    X = df.drop(target_col, axis = 1)
    y = df[target_col]
    p_train = 1#0.8

    # Train Test Split


    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    train_indices = [i for i in range(0,int(len(X)*p_train))]
    test_indices = [i for i in range(int(len(X)*p_train), len(X))]
    X_train = X.iloc[train_indices]
    #X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    #y_test = y.iloc[test_indices]


    # Scaling Data for ANN
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    #X_test = sc_X.transform(X_test)


    #num_features = len(X_train[1,:])

    # ANN with Keras
    np.random.seed(10)
    regressor = Sequential()
         # better values with tanh against relu, sigmoid...
    regressor.add(Dense(32, kernel_initializer = 'uniform',  input_dim = num_cols))#, use_bias=False))
    #regressor.add(Dropout(0.2))
    regressor.add(Dense(1, kernel_initializer = 'uniform',  input_dim = 32))

    #regressor.add(BatchNormalization(input_shape=(num_cols,)))
    regressor.add(Activation('relu'))
    opt = optimizers.Adam(learning_rate = 0.01) #Adam(learning_rate=0.01)
    #classifier.add(Dense(20, activation='relu'))
    #regressor.add(Dense(1, kernel_initializer = 'uniform',activation = 'relu'))
    print(regressor.summary())
    regressor.compile(optimizer = opt, loss = 'mean_squared_error') #loss = root_mean_squared_error) #        # metrics=['mse','mae']

    #regressor.compile(optimizer = opt, loss = 'mean_absolute_error') #loss = root_mean_squared_error) #        # metrics=['mse','mae']

    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10, min_delta = 0.01)  # ignored
    history_mse = regressor.fit(X_train, y_train, epochs = 100,
                                callbacks = [early_stopping_monitor], verbose = 0, validation_split = 0.3)

    #print('Loss:    ', history_mse.history['loss'][-1], '\nVal_loss: ', history_mse.history['val_loss'][-1])



    # EVALUATE MODEL IN THE TEST SET
    #score_mse_test = regressor.evaluate(X_test, y_test)
    #print('Test Score:', score_mse_test)

    # EVALUATE MODEL IN THE TRAIN SET
    #score_mse_train = regressor.evaluate(X_train, y_train)
    #print('Train Score:', score_mse_train)

    plt_flag=0
    if plt_flag==1:
        plt.figure(figsize=(15, 6))
        plt.plot(history_mse.history['loss'], lw =3, ls = '--', label = 'Loss')
        plt.plot(history_mse.history['val_loss'], lw =2, ls = '-', label = 'Val Loss')
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.title('RMSE')
        plt.legend()
        plt.show()

    #print('yay')
    print(regressor.get_weights())
    #Converting the first line of the dataset
    linha1 = np.array([5,5,2]).reshape(1,-1)
    # Scaling the first line to the same pattern used in the model
    linha1 = sc_X.transform(linha1)
    # Predicted value by model
    y_pred_mse_1 = regressor.predict(linha1)
    print('Predicted value: ',y_pred_mse_1)
    print('Real value: ','2.0')

    return regressor, sc_X


#df = pd.read_csv('housingdata.csv', header = None)
#approx_Lindley(df)

def pred_keras_lindley(new_si, new_ti, regressor, sc_X):


    s_prev = [0]
    s_prev.extend([new_si[i - 1] for i in range(1, len(new_si))])
    w_prev = [0]

    for i in range(len(s_prev)):
        linha1 = np.array([new_ti[i], s_prev[i], w_prev[len(w_prev)-1]]).reshape(1, -1)
        # Scaling the first line to the same pattern used in the model
        try:
            linha1 = sc_X.transform(linha1)
        except ValueError:
            print('w_prev: ', w_prev)
            return ValueError
        # Predicted value by model
        y_pred_mse_1 = regressor.predict(linha1).ravel()[0]
        if i<0:
            print('Params: ', new_ti[i], s_prev[i], w_prev[len(w_prev)-1])
            print('Predicted value: ', y_pred_mse_1)
        w_prev.append(y_pred_mse_1)

    mean_di = np.mean(w_prev)
    #print("Waiting after change: " + str(mean_di))
    var_d = pow(np.sum([pow(d - mean_di, 2) for d in w_prev]) / (len(w_prev) - 1), 0.5)
    #print("STDEV of LOS after change: " + str(var_d))

    return w_prev, var_d