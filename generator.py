import pandas as pd


from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import rv_histogram

from sklearn.tree import DecisionTreeRegressor

from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
def erlang_sample(K, m):
    # Generate K samples from exponential distribution with rate m
    exponential_samples = np.random.exponential(scale=1 / m, size=K)

    # Compute the sum of K exponential samples
    erlang_sample = np.sum(exponential_samples)

    return erlang_sample
def sample_with_replacement(residuals, num_samples):
    # Get the number of residuals
    num_residuals = len(residuals)

    # Sample indices uniformly with replacement
    sample_indices = np.random.choice(range(num_residuals), size=num_samples, replace=True)

    # Sampled residuals
    sampled_residuals = residuals[sample_indices]

    return sampled_residuals

def empty_func(x):
    return x
def return_func(exp_flag=False):
    if exp_flag:
        return np.exp
    else:
        return empty_func

def cart_cluster(X,y, min_obs):

    model = DecisionTreeRegressor(min_samples_leaf=min_obs)
    model.fit(X.values, y)
    plt.figure(figsize=(10, 6))  # Set the figure size
    plot_tree(model, feature_names=X.columns, filled=True)  # Plot the tree
    plt.savefig('decision_tree.pdf')
    plt.clf()

    return model.apply(X), model




def generate_sojourn_times(X,y, clusters, clusterer,  log_scale=True, model_based_sampling=True, allow_neg=False):

    #may want to remove first 1000 customers for warm up
    if log_scale:
        y = np.log(y)

    #X = df.iloc[:, :-1]
    #y = df.iloc[:, -1]

    seed = 42
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=seed)
    C_train = clusters.values[X_train['arr_nis'].index]
    #C_test = clusters[X_test.index]
    ebm = ExplainableBoostingRegressor()
    if model_based_sampling:
        ebm_predict=True
        if ebm_predict:

            ebm.fit(X_train, y_train)
            #ebm.fit(C_train, y_train)

            #residuals = np.array(y_train-ebm.predict(C_train))
            residuals = np.array(y_train-ebm.predict(X_train))

            # Make predictions on the test set

            #C_pred = clusterer.apply(X_test)
            #y_pred = ebm.predict(C_pred)
            y_pred = ebm.predict(X_test)


        else:
            # Create an instance of Linear Regression
            linear_regressor = LinearRegression()

            # Fit the linear regression model
            linear_regressor.fit(X_train, y_train)
            residuals = np.array(y_train-linear_regressor.predict(X_train))

            #Make predictions using linear regression
            C_pred = clusterer.apply(X_test)
            y_pred = linear_regressor.predict(C_pred)
            #y_pred = linear_regressor.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(return_func(log_scale)(y_test), return_func(log_scale)(y_pred))
        print("Mean Squared Error Predicted:", mse)

        res_sample = sample_with_replacement(residuals, num_samples=len(y_test))
        #print(res_sample)

        generated_sojourns = return_func(log_scale)(y_pred + res_sample)



        if allow_neg==False:
            count = 0

            while np.any(generated_sojourns <= 0):
                generated_sojourns = y_pred + res_sample

                count+=1
                negative_indices = np.where(generated_sojourns <= 0)[0]
                res_sample[negative_indices] = sample_with_replacement(residuals, num_samples=len(negative_indices))

            print(count, min(generated_sojourns))


        print("Variances - test vs. generated", np.var(return_func(log_scale)(y_test)), np.var(generated_sojourns))
        #mse = mean_squared_error(return_func(log_scale)(y_test), generated_sojourns)
        #print("Mean Squared Error Generated:", mse)
    else:

        # Calculate the bin size and number of bins based on the desired bin width

        #TODO:partition by nis

        bin_width = 0.001
        num_bins = int((np.max(y_train) - np.min(y_train)) / bin_width)

        # Fit empirical distribution (histogram) to the log numbers with the specified bin size
        hist, bins = np.histogram(y_train, bins=num_bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create an empirical distribution from the histogram
        empirical_dist = rv_histogram((hist, bins))

        # Sample new numbers from the empirical distribution
        generated_sojourns = return_func(log_scale)(empirical_dist.rvs(size=len(y_test)))

    bin_width = 0.5
    bin_count = int((generated_sojourns.max() - generated_sojourns.min()) / bin_width)
    # Plot the distribution of test "sojourn"
    plt.hist(return_func(log_scale)(y_test), bins=bin_count,  alpha=0.5, label='Test "sojourn"', density = True)

    # Plot the distribution of predicted "sojourn"
    #plt.hist(return_func(log_scale)(y_pred), bins=30, alpha=0.5, label='Predicted "sojourn"')
    plt.hist(generated_sojourns, bins=bin_count, alpha=0.5, label='Generated "sojourn"', density = True)

    # Add labels and legend
    plt.xlabel("Sojourn")
    plt.ylabel("Frequency")
    plt.legend()

    # Show the plot
    plt.show()
    return ebm


plt_flag=False
if plt_flag:
    df = pd.read_csv('horizontal_many.csv')
    plt.hist(df['arr_nis'], bins=50,  alpha=0.5, label='Real NIS', density = True)
    plt.show()


    fig, ax = plt.subplots()

    # Plot the first time series

    # Plot the second time series
    ax.plot(df['arr_nis'].values[0:1000], label='Actual NIS')

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('NIS')
    ax.set_title('Comparison of Generated vs. Actual')

    # Display the legend
    ax.legend()

    # Show the plot
    plt.show()


gen_run = False
if gen_run:
    df = pd.read_csv('horizontal_large.csv')
    #{'case', 'class':[], 'arrival':[], 'service':[], 'departure':[], 'server':[], 'arr_nis':[], 'arr_niq':[], 'sojourn'}
    X = df[["arr_nis"]]
    y = df['sojourn']
    df['cluster'], clusterer = cart_cluster(X, y, min_obs=100)
    #print(df['cluster'])
    clusters =df[['cluster']]
    #X = df[['cluster']]

    ebm = generate_sojourn_times(X, y, clusters, clusterer,  log_scale=True,
                           model_based_sampling=True, allow_neg=True)

