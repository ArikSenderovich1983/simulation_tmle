import torch # For building the networks
import torchtuples as tt # Some useful functions
_ = torch.manual_seed(123)
# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from pycox.datasets import metabric
from pycox.models import PMF
from pycox.evaluation import EvalSurv



def mg1_speedup_pycox_service_model(df, target_col, num_cols):

    #assuming the following features:
    #target variable is service, the rest are features.

    print(df.head())
    print(len(df))
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    p_train = 1  # 0.8

    # Train Test Split

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    train_indices = [i for i in range(0, int(len(X) * p_train))]
    test_indices = [i for i in range(int(len(X) * p_train), len(X))]
    X_train = X.iloc[train_indices]
    # X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    # y_test = y.iloc[test_indices]




