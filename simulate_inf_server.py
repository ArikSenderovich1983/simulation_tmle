import pandas as pd
#import generator as gn
import heapq
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from scipy.stats import gamma
from scipy.special import kl_div
from scipy.stats import entropy
from pre_processing import *
import warnings
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
warnings.filterwarnings("ignore")

#suppl. functions
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

#plotting and stats
def plot_hist(generated_series, actual_series, x_label, num_bin):
    #plotting a histogram of two series, x axis label, and num bins
    #generated_hist, generated_bins = np.histogram(generated_series, bins=num_bin, density=True)
    #actual_hist, actual_bins = np.histogram(actual_series, bins=num_bin, density=True)
    #bin_width = generated_bins[1] - generated_bins[0]
    #plt.bar(generated_bins[:-1], generated_hist, width=bin_width, alpha=0.5, label='Generated '+str(x_label), align='edge')
    #plt.bar(actual_bins[:-1], actual_hist, width=bin_width, alpha=0.5, label='Real '+str(x_label), align='edge')

    generated_hist, generated_bins, _ = plt.hist(generated_series, bins=num_bin, density=True, alpha=0.5,
                                                 label='Generated ' + str(x_label))
    actual_hist, actual_bins, _ = plt.hist(actual_series, bins=num_bin, density=True, alpha=0.5,
                                           label='Real ' + str(x_label))

    #plt.hist(generated_series, num_bins  = )


    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.title('Histogram of '+x_label)
    plt.show()
def plot_ts(gen_x_values,  generated_series, actual_series, x_label):
    fig, ax = plt.subplots()

    # Plot the first time series
    ax.plot(gen_x_values, generated_series, label='Generated '+x_label)

    # Plot the second time series
    ax.plot(gen_x_values, actual_series, label='Actual '+x_label)

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel(x_label)
    ax.set_title('Comparison of Generated vs. Actual '+x_label)

    # Display the legend
    ax.legend()

    # Show the plot
    plt.show()
def ks_test(generated_series, actual_series, x_label):
    statistic, p_value = stats.ks_2samp(generated_series, actual_series)
    # Display the test result
    print("Statistic: "+x_label, statistic)
    print("P-value: "+x_label, p_value)
def compute_stats(generated_log, df):

    plot_hist(generated_log['s_arr_nis'], df['arr_nis'], x_label="NIS", num_bin=50)
    plot_hist(generated_log['s_sojourn'], df['sojourn'], x_label="Sojourn", num_bin=50)

    # Define the bin intervals
    bin_intervals = np.arange(0, max(generated_log['arrival']), 0.0001)
    # Bin the time series data
    x_gen_arrivals = np.digitize(generated_log['arrival'], bin_intervals) - 1

    plot_ts(x_gen_arrivals, #df['arrival'].values[0:len(generated_log['arrival'])],
            generated_log['s_arr_nis'], df['arr_nis'].values[0:len(generated_log['arrival'])], "NIS")
    plot_ts(x_gen_arrivals, #df['arrival'].values[0:len(generated_log['arrival'])],
           generated_log['arrival_count'], df['cum_arrival'].values[0:len(generated_log['arrival'])],
            "Cumulative Arrival")

    #plot_ts(generated_log['arrival'], df['arrival'].values[0:len(generated_log['arrival'])],  generated_log['nis_arrival'], df['arr_nis'].values[0:len(generated_log['arrival'])], "NIS")
    #plot_ts(generated_log['arrival'], df['arrival'].values[0:len(generated_log['arrival'])], generated_log['arrival_count'], df['cum_arrival'].values[0:len(generated_log['arrival'])], "Cumulative Arrival")

    ks_test(generated_log['s_sojourn'], df['sojourn'].values[0:len(generated_log['arrival'])], "Sojourn")
    ks_test(generated_log['s_arr_nis'], df['arr_nis'].values[0:len(generated_log['arrival'])], "NIS")

#learning and bootstrapping
def learn_generator(X, y , log_scale=True, model_based_sampling=True, allow_neg=False):
        if log_scale:
            y = np.log(y)
        ebm = ExplainableBoostingRegressor()
        residuals = []
        if model_based_sampling:
            ebm_predict = True
            if ebm_predict:
                #ebm.fit(clusters, y)
                #residuals = np.array(y - ebm.predict(clusters))
                ebm.fit(X, y)
                residuals = np.array(y - ebm.predict(X))
        return ebm, residuals
def sample_arrivals(df, static_context, n):
    # we will bootstrap arrivals
    class_counts = df['class'].value_counts()
    # Compute the proportions by dividing the counts by the total number of samples
    class_proportions = class_counts / len(df)
    # Convert the proportions to a dictionary
    class_values = (class_proportions * n).round().astype(int)
    diff = n - class_values.sum()
    class_values[class_values.idxmax()] += diff
    # Convert the values to a dictionary
    class_values_dict = class_values.to_dict()


    # Optional: Shuffle the rows in the new DataFrame
    sampled_context = df[static_context].sample(n=n, replace=True).reset_index(drop=True)


    #todo: idea - sample from same class, but sample indices to grab context.
    sampled_arrivals = {}
    for c in df['class'].unique():
        df_arrivals = df[df['class']==c]['arrival'].copy(deep=True)
        df_arrivals.sort_values(inplace=True)
        df_arrivals.reset_index(inplace=True, drop=True)
        time_between_arrivals = df_arrivals.diff()
        time_between_arrivals = time_between_arrivals.drop(0)
        sampled_arrivals[c] = sample_with_replacement(time_between_arrivals.values, class_values_dict[c])

    return sampled_arrivals, sampled_context
def compare_cluster_distributions(X, y, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_

    gamma_params = []
    for cluster_id in range(k):
        cluster_data = y[cluster_labels == cluster_id]
        shape, loc, scale = gamma.fit(cluster_data, floc=0)
        gamma_params.append((shape, loc, scale))

    # Compute similarity score based on KL divergence
    similarity_score = 0.0
    num_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            x = np.linspace(1, 10, 1000)

            # Calculating the PDFs of the gamma distributions
            pdf1 = gamma.pdf(x, *gamma_params[i])
            pdf2 = gamma.pdf(x, *gamma_params[j])

            # Calculating the KL divergence using scipy.stats.entropy
            kl_divergence = entropy(pdf1, pdf2)
            similarity_score += kl_divergence
            num_pairs += 1

    if num_pairs > 0:
        similarity_score /= num_pairs
    return similarity_score
def find_best_k(X, y, k_range):
    best_score = -1
    best_k = -1
    count = 0
    for k in k_range:
        if count>3:
            break
        print('k is ', k)
        score = compare_cluster_distributions(X, y, k)
        if score > best_score:
            best_score = score
            best_k = k
            count = 0
        else:
            count+=1
    print('Best k is ', best_k)
    return best_k
def get_arrivals(df):
    sampled_arrivals = {}
    for c in df['class'].unique():
        df_arrivals = df[df['class']==c]['arrival'].copy(deep=True)
        df_arrivals.sort_values(inplace=True)
        df_arrivals.reset_index(inplace=True, drop=True)
        time_between_arrivals = df_arrivals.diff()
        time_between_arrivals = time_between_arrivals.drop(0)
        sampled_arrivals[c] = time_between_arrivals.values
        # print("time between arrivals:  ", time_between_arrivals.values)
        print()
    return sampled_arrivals

def get_arrivals2(df):
    sampled_arrivals = {}
    df_arrivals = df['arrival'].copy(deep=True)
    df_arrivals.sort_values(inplace=True)
    df_arrivals.reset_index(inplace=True, drop=True)
    time_between_arrivals = df_arrivals.diff()
    time_between_arrivals = time_between_arrivals.drop(0)
    sampled_arrivals = time_between_arrivals.values
    sampled_arrivals = np.append(0, sampled_arrivals)
    # print("time between arrivals:  ", time_between_arrivals.values)
    print()
    return sampled_arrivals

def fit_gamma(df, X, y_label,K):
    kmeans = KMeans(n_clusters=K)  # Adjust the number of clusters as per your requirement
    kmeans.fit(X.values)
    df['cluster'] = kmeans.labels_
    gamma_params = []
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id][y_label].values
        shape, loc, scale = gamma.fit(cluster_data, floc=0)  # Assuming durations are positive, hence setting floc=0
        gamma_params.append((shape, loc, scale))
    return kmeans, gamma_params

def sample_from_gamma_clusters(num_samples, shape, loc, scale):
    samples = gamma.rvs(shape, loc=loc, scale=scale, size=num_samples)
    return samples

#simulation functions:
def write_event(nis, generator, residuals, departure_calendar,
                log_scale, generated_log, arrival_count,
                kmeans, gamma_params, generator_flag, nis_vec, arr_timestamp, new_class, prev_arrival, prev_sojourn, context, last_los_vec, prev_class, col_names):

    arrival_count += 1
    nis_vec[new_class] += 1
    nis = sum(nis_vec)


    #an arrival event
    #write the event down
    generated_log['arrival'].append(arr_timestamp)
    last_los = max(0,prev_sojourn - (arr_timestamp-prev_arrival))
    if prev_class == 0:
        last_los_vec[0] = max(0, prev_arrival + prev_sojourn - arr_timestamp)
        last_los_vec[1] = max(0, last_los_vec[1] - arr_timestamp + prev_arrival)
        last_los_vec[2] = max(0, last_los_vec[2] - arr_timestamp + prev_arrival)
    elif prev_class == 1:
        last_los_vec[1] = max(0, prev_arrival + prev_sojourn - arr_timestamp)
        last_los_vec[0] = max(0, last_los_vec[0] - arr_timestamp + prev_arrival)
        last_los_vec[2] = max(0, last_los_vec[2] - arr_timestamp + prev_arrival)
    elif prev_class == 2:
        last_los_vec[2] = max(0, prev_arrival + prev_sojourn - arr_timestamp)
        last_los_vec[0] = max(0, last_los_vec[0] - arr_timestamp + prev_arrival)
        last_los_vec[1] = max(0, last_los_vec[1] - arr_timestamp + prev_arrival)
    # print("context:", context)
    # pred_features = [nis, new_class, last_los]
    pred_features = []
    pred_features.extend(context)
    if "arr_nis" in col_names:
        ind = col_names.index("arr_nis")
        pred_features[ind] = nis
    if "last_los_remaining" in col_names:
        ind = col_names.index("last_los_remaining")
        pred_features[ind] = last_los
    if "arr_nis_class_0" in col_names:
        for c in all_classes:
            ind = col_names.index("arr_nis_class_0")
            pred_features[ind+c] = nis_vec[c]
    if "last_rem_class_0" in col_names:
        for c in all_classes:
            ind = col_names.index("last_rem_class_0")
            pred_features[ind+c] = nis_vec[c]

    if generator_flag:
        gen_duration = [0]
        while gen_duration[0]<=0:
            gen_duration = return_func(log_scale)(generator[new_class].predict([pred_features]) + sample_with_replacement(residuals[new_class], num_samples=1))
    else:
        cur_params = gamma_params[kmeans.predict([pred_features])[0]]
        gen_duration = sample_from_gamma_clusters(1, cur_params[0], cur_params[1], cur_params[2])
        
    
    '''
    if consult:
        gen_duration = max(gen_duration - 30, 5)
    '''


    departure_timestamp = arr_timestamp + gen_duration
    #departure time is added to the calendar
    heapq.heappush(departure_calendar, (departure_timestamp[0], new_class, context))
    #departure and sojourn times documented
    generated_log['s_departure'].append(departure_timestamp[0])
    generated_log['s_sojourn'].append(departure_timestamp[0] - arr_timestamp)
    #nis upon arrival
    generated_log['s_arr_nis'].append(nis)
    generated_log['s_last_remaining_los'].append(last_los)
    for c in all_classes:
        generated_log['s_arr_nis_class_'+str(c)].append(nis_vec[c])
        generated_log['s_last_rem_class_'+str(c)].append(last_los_vec[c])
    generated_log['arrival_count'].append(arrival_count)
    generated_log['class'].append(new_class)
    prev_sojourn = gen_duration[0]
    prev_arrival = arr_timestamp
    

    return prev_sojourn, prev_arrival, arrival_count, nis, nis_vec

def simulate_system(generator, residuals, log_scale, sampled_arrivals, sampled_context, kmeans, gamma_parms, generator_flag, all_classes, classes, col_names):
    # heapq.heapify(arrival_calendar)
    arrival_calendar = []
    departure_calendar = []


    # print("sampled context", sampled_context)

    cur_ts = 0
    eps_ = 0.00001
    all_ts_ties = []
    count_ties = 0
    count_for_context = 0

    for s in sampled_arrivals: 
        cur_ts += s
        while cur_ts in all_ts_ties:
            #tie break
            cur_ts = cur_ts + eps_
            count_ties+=1
        heapq.heappush(arrival_calendar, (cur_ts, classes[count_for_context], sampled_context.iloc[count_for_context].values))
        all_ts_ties.append(cur_ts)
        count_for_context += 1
    
    print('Ties:', count_ties)
    nis = 1
    nis_vec = []
    last_los_vec = []
    generated_log = {'arrival_count':[], 'class': [], 'arrival': [], 's_departure': [], 's_arr_nis': [], 's_sojourn': []}
    for c in all_classes:
        last_los_vec.append(0)
        # if c == classes[0]:
        #     nis_vec.append(1)
        # else:
        nis_vec.append(0)
        generated_log['s_arr_nis_class_'+str(c)] = []
    generated_log['s_last_remaining_los'] = []
    for c in all_classes:
        generated_log['s_last_rem_class_' + str(c)] = []
    arrival_count = 0
    

    #first arrival
    arr_timestamp, new_class, context = heapq.heappop(arrival_calendar)
    prev_arrival = arr_timestamp
    prev_sojourn = 0
    prev_class = new_class


    # print("first arrival", arr_timestamp, new_class, context)

    prev_sojourn, prev_arrival, arrival_count, nis, nis_vec = write_event(nis, generator, residuals, departure_calendar, log_scale, generated_log,
                    arrival_count, kmeans, gamma_parms, generator_flag, nis_vec, arr_timestamp, new_class, prev_arrival, prev_sojourn, context, last_los_vec, prev_class, col_names)


    while len(arrival_calendar) > 0:
        if arrival_count % 500 == 0:
            print('arrivals # :', arrival_count)
            print('nis vec: ', nis_vec)
            print('total nis:', nis)
            print('last los vec:', last_los_vec)

        if len(departure_calendar)==0:
            '''
            line below didnt have , context ?
            '''
            arr_timestamp, new_class, context = heapq.heappop(arrival_calendar)
            prev_sojourn, prev_arrival, arrival_count, nis, nis_vec = \
                write_event(nis, generator, residuals, departure_calendar,
                            log_scale, generated_log, arrival_count, kmeans, gamma_parms, generator_flag, nis_vec,
                            arr_timestamp, new_class, prev_arrival, prev_sojourn, context, last_los_vec, prev_class, col_names)
            prev_class = new_class

        elif arrival_calendar[0] <= departure_calendar[0]:

            arr_timestamp, new_class, context = heapq.heappop(arrival_calendar)
            prev_sojourn, prev_arrival, arrival_count, nis, nis_vec =\
                write_event(nis,  generator, residuals, departure_calendar,
                            log_scale, generated_log, arrival_count, kmeans, gamma_parms, generator_flag, nis_vec, 
                            arr_timestamp, new_class, prev_arrival, prev_sojourn, context, last_los_vec, prev_class, col_names)
            prev_class = new_class

        else:
            _, new_class, _ = heapq.heappop(departure_calendar)
            nis_vec[new_class]-=1
            nis = sum(nis_vec)

    while len(departure_calendar) > 0:
        _, _, _ = heapq.heappop(departure_calendar)
        nis_vec[new_class] -= 1
        nis = sum(nis_vec)

    print('arr calendar: ', len(arrival_calendar))
    print('dep calendar: ', len(departure_calendar))

    return generated_log

def plot_graphs(sojourn, s_sojourn, name, class_list, all_classes):
    print(len(sojourn), len(s_sojourn), len(class_list))
    new_d = {"sojourn": sojourn, "s_sojourn": s_sojourn, "class": class_list}
    new_df = pd.DataFrame(data=new_d)
    [sojourn, s_sojourn, classes]
    for c in all_classes:
        y = new_df[new_df["class"] == c]["sojourn"]
        y2 = new_df[new_df["class"] == c]["s_sojourn"]
        qq_plot(y, y2, name + "-Class " + str(c))


def qq_plot(data1, data2, name, plot_min=0, plot_max=None):
    """
    Plots the Q-Q plots of quantiles of actual and simulated LOS for 1 patient type and saves the plot

    @ params:
        data1 (list): quantiles from data on the x-axis --> actual LOS
        data2 (list): quantiles from data on the y-axis --> simulated LOS
        plot_min (int, default=None): minimum value for x-axis and y-axis
        plot_max (int, default=None): maximum value for x-axis and y-axis
    """

    # Set up the plot parameters
    sns.set_color_codes("colorblind")
    fig = None
    fig, _ = plt.subplots(1, 1, figsize=(8, 6), sharex='all', sharey='all', clear=True)  # ((ax1))
    
    plot_name = 'Q-Q Plot ' + str(name)
    fig.suptitle(plot_name)

    for i, ax_n in enumerate(fig.get_axes()):
        # Computes quantiles and plots them
        quantiles = min(len(data1), len(data2))
        quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
        x_quantiles = np.quantile(data1, quantiles)
        y_quantiles = np.quantile(data2, quantiles)
        ax_n.scatter(x_quantiles, y_quantiles)

        # Finds the max and min values to use for the x-axis and y-axis
        data1_min, data1_max = min(data1), max(data1)
        data2_min, data2_max = min(data2), max(data2)
        x_min = np.floor(min(data1_min, data2_min))
        x_max = np.ceil(max(data1_max, data2_max))
        x = np.linspace(x_min, x_max, int(x_max - x_min + 1))
        ax_n.plot(x, x, 'k-')  # Plots a 45-degree line

        ax_n.set_xlabel('Simulated LOS Quantiles')
        ax_n.set_ylabel('Actual LOS Quantiles')

        # Sets the x-axis limits and y-axis limits
        if (plot_min is not None) and (plot_max is not None):
            ax_n.set_xlim(plot_min, plot_max)
            ax_n.set_ylim(plot_min, plot_max)
        else:
            ax_n.set_xlim(x_min, x_max)
            ax_n.set_ylim(x_min, x_max)

    save_path = os.path.join(os.getcwd(), "{}.png".format(plot_name))
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def random_forest(X, all_classes, df, classes):
    rfs = []
    residuals = []
    X["class"] = classes
    X_dfs = []
    for c in all_classes:
        rfs.append(RandomForestRegressor(min_samples_leaf=30, max_depth=None, n_estimators=100, random_state=0))
        X_dfs.append(X[X["class"] == c])
        y = df[df["class"] == c]["sojourn"]
        X_dfs[c].drop("class", axis = 1, inplace = True)
        rfs[c].fit(X_dfs[c], y)
        residuals.append(np.array(y - rfs[c].predict(X_dfs[c])))

    X.drop("class", axis = 1, inplace = True)

    return rfs, residuals


if __name__ == "__main__":
    nis_residual_specs = 3
    # nis_residual_specs: 
    # 0 = arr_nis + last_rem_los
    # 1 = arr_nis per class + last_rem_los
    # 2 = arr_nis + last_rem_los per class
    # 3 = arr_nis per class + last_rem_los per class

    #initialize the experiment:
    n = 1000
    # df = pd.read_csv('horizontal_multi_many.csv')
    pre_processed = True
    if pre_processed:
        df = pd.read_pickle("1000_original_cleaned.pkl")
    else:
        df = main_process("10500_original.csv")
    
    print("dataframe loaded")
    #todo:
    #1. Arik to check log scale issue.
    #2. Daphne: use data arrivals for now
    #3. Daphne: one-hot encode "class"
    #4. Run NYGH data through this pipeline
    #5. EBM - feature importance
    static_context = ['class', 'Triage Code', 'Age Category', 'Initial Zone', 'Gender Code', 'Ambulance', 'Consult', 'Admission']# ['x1', 'x2']
    '''
    dummie_context = []
    for con in static_context:
        for item in df[con].unique():
            dummie_context.append(con+"_" + str(item))
    print("dummie_context:", dummie_context)
    '''
    # static_context = []
    features = []
    # dont give arr_nis ? 
    for s in static_context:
        features.append(s)
    all_classes = df['class'].astype(int).unique().tolist() # make 0, 1, 2
    all_classes.sort()
    
    match nis_residual_specs:
        case 0:
            features.append("arr_nis")
            features.append("last_remaining_los")
        case 1:
            for c in all_classes:
                features.append('arr_nis_class_' + str(c))
            features.append("last_remaining_los")
        case 2:
            features.append("arr_nis")
            for c in all_classes:
                features.append('last_rem_class_' + str(c))
        case 3:
            for c in all_classes:
                features.append('arr_nis_class_' + str(c))
            for c in all_classes:
                features.append('last_rem_class_' + str(c))


    X = df[features]
    X = pd.get_dummies(data=X, drop_first=True)
    classes = df['class'].copy(deep=True).astype(int).to_numpy()

    col_names = X.columns.tolist()
    print(col_names, len(X.columns))
    y = df['sojourn']
    log_scale = False
    generator_flag = True
    ebm_generators = []
    ebm_residuals = []
    ebm_generator, ebm_residual = learn_generator(X, y, log_scale=log_scale, model_based_sampling=True, allow_neg=True)
    rf_generators, rf_residuals = random_forest(X, all_classes, df, classes)
    for c in all_classes:
        ebm_generators.append(ebm_generator)
        ebm_residuals.append(ebm_residual)
    generators = [ebm_generators, rf_generators, False]
    residuals = [ebm_residuals, rf_residuals, False]
    generator_names = ["EBM", "RF", "Clustering"]

    # replace X with train in learn_generator
    K_max = 20
    k_best = find_best_k(X, y, range(2,K_max))
    kmeans, gamma_params = fit_gamma(df, X, y_label='sojourn', K=k_best)
    # bootstrap n new arrivals (new horizon)
    runs = 1
    for r in range(runs):
        # sampled_arrivals, sampled_context = sample_arrivals(df, static_context,  n)
        
        # sampled_arrivals = df['arrival'].diff().copy()#copy()#sample_arrivals(df, n)
        # sampled_arrivals = sampled_arrivals.drop(0)
        # sampled_arrivals = sampled_arrivals.values[0:n]
        
        sampled_arrivals = get_arrivals2(df)
        sampled_context = X
        # here still on train
        # print("sampled arrivals keys:", sampled_arrivals.keys())
        # print("sampled_context:", sampled_context)

        # print("sampled arrivals:", sampled_arrivals)
        
        
        #simulate the qnet
        for i in range(3):
            if generators[i] == False and residuals[i] == False:
                # clustering
                generator_flag = False
            start = timer()
            generated_log = simulate_system(generators[i], residuals[i], log_scale,
                                        sampled_arrivals, sampled_context,  kmeans, gamma_params, generator_flag, all_classes, classes, col_names)
            
            end = timer()
            print("time for run:", end-start)
            #write results
            pd.DataFrame.from_dict(generated_log).to_csv('results_'+ str(generator_names[i]) + "_" +str(r)+'.csv', index=False)
            print("saved results")
            # compute and plot stats
            for k, v in generated_log.items():
                generated_log[k] = v[100:]
            # compute_stats(generated_log, df.iloc[100:])
            print("calculated stats")
            data1 = y.to_numpy()[100:]
            data2 = generated_log['s_sojourn']
            plot_graphs(data1, data2, generator_names[i], classes[100:], all_classes)
            # qq_plot(data1, data2, generator_names[i])
            

            # choose best test out of the different runs (8 options)
            # run simulate sysyem on the test data with the "best" moddel
