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
from datetime import timedelta
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
def plot_hist(generated_series, actual_series, x_label, num_bin, name):
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

    plt_title = 'Histogram of '+ x_label
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.title(plt_title)
    save_path = os.path.join(os.getcwd(), "{}.png".format(plt_title + "-" + name))
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_ts(gen_x_values,  generated_series, actual_series, x_label, name, spec_name, time_name):
    '''
    Graphs a time series graph for a given value
    params: gen_x_values: x values for the time
    generated_series: actual data graphed over time (from simulation)
    actual series: data graphed over time (actual)
    x_label: name for the graph
    spec_name: name of specification
    time_name: training or testing months
    '''
    fig = None
    fig, ax = plt.subplots()

    # Plot the first time series
    ax.plot(gen_x_values, generated_series, label='Generated '+x_label)

    # Plot the second time series
    ax.plot(gen_x_values, actual_series, label='Actual '+x_label)

    plt.xlabel("Time")
    plt.ylabel(x_label)
    plt_title = time_name + ' Comparison of Generated vs. Actual '+ x_label
    plt.title(plt_title)

    # Display the legend
    ax.legend()

    filename = time_name + " " + spec_name + ' Time-Series ' + x_label + '-' + name
    save_path = os.path.join(os.getcwd(), "{}.png".format(filename))
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def ks_test(generated_series, actual_series, x_label):
    statistic, p_value = stats.ks_2samp(generated_series, actual_series)
    # Display the test result
    # print("Statistic: "+x_label, statistic)
    # print("P-value: "+x_label, p_value)
    return p_value, statistic

def plot_graphs(class_dfs_per_gen, generator_names, class_names, spec_name, time_name):
    '''
    plots qq plots and box whisker plots
    '''
    data_s_sojourn = []
    data_nis = []
    sojourn = []
    # loop for each class and generator to create data lists for nis and sojourn
    # lists are then used to create class specific graphs
    for c in range(len(class_names)):
        for gen in range(len(generator_names)):
            arr_nis = "arr_nis"
            s_arr_nis = "s_arr_nis"
            if c != 0:
                arr_nis = arr_nis + "_class_" + str(c-1)
                s_arr_nis = s_arr_nis + "_class_" + str(c-1)
            df_to_use = class_dfs_per_gen[gen][c]
            if gen == 0:
                sojourn = df_to_use["sojourn"]
                data_nis.append(df_to_use[arr_nis])
            data_s_sojourn.append(df_to_use['s_sojourn'])
            data_nis.append(df_to_use[s_arr_nis])
        qq_plot(sojourn, data_s_sojourn, class_names[c], generator_names, spec_name, time_name)
        data = [sojourn]
        data.extend(data_s_sojourn)
        plot_box_whisker(data, class_names[c], generator_names, "Sojourn", spec_name, time_name)
        plot_box_whisker(data_nis, class_names[c], generator_names, "NIS", spec_name, time_name)
        data_s_sojourn = []
        data_nis = []

def plot_box_whisker(data_list, class_name, generator_names, name, spec_name, time_name):
    '''
    Graphs box whisker plots given a list of data sets
    '''
    plt_title = time_name + " " + spec_name + ' Box Plot '+ name + "-" + class_name
    fig, ax = plt.subplots()
    fig.suptitle(plt_title)
    elements = []
    labels = ["Real " + name]
    labels.extend(generator_names)
    colours = ['blue', 'red', 'green', 'orange']
    for d, data in enumerate(data_list):
        # graphs all box_plots for different generators but the same cclass on class specific pngs
        elements.append(ax.boxplot(data, positions=[ 1 + 3*d], widths = [2], patch_artist=True, boxprops=dict(facecolor=colours[d])))
   
    ax.legend([element["boxes"][0] for element in elements], [labels[idx] for idx,_ in enumerate(data_list)], loc='upper right')

    filename = plt_title
    save_path = os.path.join(os.getcwd(), "{}.png".format(filename))
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_intervention_box_plots(data_list, class_name, int_names, time_name):
    '''
    plots box plots from interventions
    '''
    plt_title = time_name + " Box Plot-" + class_name
    fig, ax = plt.subplots()
    fig.suptitle(plt_title)
    elements = []
    labels = int_names
    colours = ['blue', 'red', 'green', 'orange']
    for d, data in enumerate(data_list):
        # box plots have each of the four interventions for a specific class
        elements.append(ax.boxplot(data["s_sojourn"], positions=[1+3*d], widths = [2], patch_artist=True, boxprops=dict(facecolor=colours[d])))
   
    ax.legend([element["boxes"][0] for element in elements], [labels[idx] for idx,_ in enumerate(data_list)], loc='upper right')

    # fix naming system
    plt.savefig('Boxplots-{}.png'.format(class_name), dpi=300)
    # plt.show()
    plt.close()

def prepare_box_plots(class_spec_dfs, intervention_names, class_names, time_name):
    '''
    prepares the data for the box plots
    created data lists of the different results per intervention for a certain class
    '''
    data = []
    for c, name in enumerate(class_names):
        for intrv in intervention_names:
            data.append(class_spec_dfs[intrv][c])
        plot_intervention_box_plots(data, name, intervention_names, time_name)
        data = []

def qq_plot(data1, data2, name, generator_names, spec_name, time_name, plot_min=0, plot_max=None):
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
    
    plot_name = time_name + " " + spec_name + 'Q-Q Plot ' + str(name)
    fig.suptitle(plot_name)

    for i, ax_n in enumerate(fig.get_axes()):
        for count in range(len(data2)):    
            # Computes quantiles and plots them
            quantiles = min(len(data1), len(data2[count]))
            quantiles = np.linspace(start=0, stop=1, num=int(quantiles))
            x_quantiles = np.quantile(data1, quantiles)
            y_quantiles = np.quantile(data2[count], quantiles)
            ax_n.scatter(x_quantiles, y_quantiles, label=generator_names[count])


        # Finds the max and min values to use for the x-axis and y-axis
        data1_min, data1_max = min(data1), max(data1)
        data2_min, data2_max = min(data2[0]), max(data2[0])
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

    ax_n.legend()
    save_path = os.path.join(os.getcwd(), "{}.png".format(plot_name))
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def get_ks_test_vals(dfs, class_types):
    '''
    helper function to get the statistics from the ks_test
    '''
    count_for_names = 0
    p_sojourn = []
    stat_sojourn = []
    p_nis = []
    stat_nis = []
    for class_df in dfs:
        p_soj, stat_soj = ks_test(class_df['s_sojourn'], class_df['sojourn'].values[0:len(class_df['arrival'])], "Sojourn" + "-" + class_types[count_for_names])
        p_nis_val, stat_nis_val = ks_test(class_df['s_arr_nis'], class_df['arr_nis'].values[0:len(class_df['arrival'])], "NIS" + "-" + class_types[count_for_names])
        p_sojourn.append(p_soj)
        stat_sojourn.append(stat_soj)
        p_nis.append(p_nis_val)
        stat_nis.append(stat_nis_val)
        # print(class_types[count_for_names] + "-values: ", p_nis_val, p_soj, stat_nis_val, stat_soj)
        count_for_names += 1

    return p_nis, p_sojourn, stat_nis, stat_sojourn

def plot_nis_ts(class_dfs_per_gen, generator_names, class_names, spec_name, time_name):
    '''
    plots the time series graph for the nis
    '''
    for count in range(len(class_names)):
        fig = None
        fig, ax = plt.subplots()
        
        # for each generator, plot the simulated sojourn
        for gen in range(len(generator_names)):
            arr_nis = "arr_nis"
            s_arr_nis = "s_arr_nis"
            df_to_use = class_dfs_per_gen[gen][count]
            # print(gen, count)
            # print("check:", df_to_use["s_arr_nis"])
            bin_intervals = np.arange(0, max(df_to_use['arrival']), 0.0001)
            # Bin the time series data
            x_gen_arrivals = np.digitize(df_to_use['arrival'], bin_intervals) - 1
            # Plot the actual series
            if count != 0:
                arr_nis = arr_nis + "_class_" + str(count-1)
                s_arr_nis = s_arr_nis + "_class_" + str(count-1)
            if gen == 0:
                # graph the actual nis one time
                ax.plot(x_gen_arrivals, df_to_use[arr_nis], label='Actual NIS')
            # Plot the generated series
            ax.plot(x_gen_arrivals, df_to_use[s_arr_nis], label=generator_names[gen] + '-Generated')
        
        plt.xlabel("Time")
        plt.ylabel("NIS")
        plt_title = time_name + ' Comparison of Generated vs. Actual NIS-' + class_names[count]
        plt.title(plt_title)

        # Display the legend
        ax.legend()

        filename = time_name + " " + spec_name + ' Time Series NIS'  + '-' + class_names[count]
        save_path = os.path.join(os.getcwd(), "{}.png".format(filename))
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

def graph_hist_cumulative_ts(new_dfs, gen_name, simulation_names, spec_name, time_name):
    count_for_names = 0
    for class_df in new_dfs:
        name = gen_name + "-" + simulation_names[count_for_names]
        # Define the bin intervals
        bin_intervals = np.arange(0, max(class_df['arrival']), 0.0001)
        # Bin the time series data
        x_gen_arrivals = np.digitize(class_df['arrival'], bin_intervals) - 1
        # plot the cumulative arrival to see that they are the same
        if simulation_names[count_for_names] == "All Classes":
            plot_ts(x_gen_arrivals, #df['arrival'].values[0:len(generated_log['arrival'])],
                class_df['arrival_count'], class_df['cum_arrival'].values[0:len(class_df['arrival'])],
                    "Cumulative Arrival", name, spec_name, time_name)
        count_for_names += 1

#learning and bootstrapping
def get_generators(X, training_df, all_classes, classes):
    # gets the generators for EBM and RF and returns them
    ebm_generators, ebm_residuals = ebm_models(X, all_classes, training_df, classes)
    rf_generators, rf_residuals = random_forest(X, all_classes, training_df, classes)
    generators = [ebm_generators, rf_generators, False]
    residuals = [ebm_residuals, rf_residuals, False]   
    return generators, residuals

def ebm_models(X, all_classes, df, classes):
    # creates three different ebm models for each class
    ebms = []
    residuals = []
    X["class"] = classes
    X_dfs = []
    for c in all_classes:
        ebms.append(ExplainableBoostingRegressor())
        # looks at only a certain class type
        X_dfs.append(X[X["class"] == c])
        y = df[df["class"] == c]["sojourn"]
        # drops the class column so that it isn't used in the simulation
        X_dfs[c].drop("class", axis = 1, inplace = True)
        ebms[c].fit(X_dfs[c], y)
        residuals.append(np.array(y - ebms[c].predict(X_dfs[c])))

    X.drop("class", axis = 1, inplace = True)

    return ebms, residuals

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

def get_arrivals(df_for_use):
    # returns a list of all the arrival values
    sampled_arrivals = {}
    df_arrivals = df_for_use['arrival'].copy(deep=True)
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

def random_forest(X, all_classes, df, classes):
    # creates three different RF models (one for each class)
    rfs = []
    residuals = []
    X["class"] = classes
    X_dfs = []
    for c in all_classes:
        rfs.append(RandomForestRegressor(min_samples_leaf=30, max_depth=None, n_estimators=100, random_state=0))
        X_dfs.append(X[X["class"] == c])
        # adds a part of the df that is a certain class
        y = df[df["class"] == c]["sojourn"]
        # drops class column so it isn't used for fitting the models
        X_dfs[c].drop("class", axis = 1, inplace = True)
        rfs[c].fit(X_dfs[c], y)
        residuals.append(np.array(y - rfs[c].predict(X_dfs[c])))

    X.drop("class", axis = 1, inplace = True)

    return rfs, residuals

#simulation functions:
def write_event(nis, generator, residuals, departure_calendar,
                log_scale, generated_log, arrival_count,
                kmeans, gamma_params, generator_flag, nis_vec, arr_timestamp, new_class, prev_arrival, prev_sojourn, context, last_los_vec, prev_class, col_names, intervention):

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

    col_for_consult = col_names.index("Consult_Yes")

    if generator_flag:
        gen_duration = [0]
        while gen_duration[0]<=0:
            if len(generator) < 3:
                print("in here")
                print(generator)
            if new_class != 0 and new_class != 1 and new_class != 2:
                print("hey")
                print(generator)
                print(new_class)
            gen_duration = return_func(log_scale)(generator[new_class].predict([pred_features]) + sample_with_replacement(residuals[new_class], num_samples=1))
    else:
        cur_params = gamma_params[kmeans.predict([pred_features])[0]]
        gen_duration = sample_from_gamma_clusters(1, cur_params[0], cur_params[1], cur_params[2])
    
    if context[col_for_consult] and gen_duration > 60 and intervention != 0:
                gen_duration = [max(60, gen_duration[0] - intervention)]
                departure_timestamp = arr_timestamp + gen_duration
        
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

def simulate_system(generator, residuals, log_scale, sampled_arrivals, sampled_context, kmeans, gamma_parms, generator_flag, all_classes, classes, col_names, intervention):
    # heapq.heapify(arrival_calendar)
    arrival_calendar = []
    departure_calendar = []

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
                    arrival_count, kmeans, gamma_parms, generator_flag, nis_vec, arr_timestamp, new_class, prev_arrival, prev_sojourn, context, last_los_vec, prev_class, col_names, intervention)


    while len(arrival_calendar) > 0:
        # if arrival_count % 500 == 0:
            # print('arrivals # :', arrival_count)
            # print('nis vec: ', nis_vec)
            # print('total nis:', nis)
            # print('last los vec:', last_los_vec)

        
        if len(departure_calendar)==0:
            '''
            line below didnt have , context ?
            '''
            arr_timestamp, new_class, context = heapq.heappop(arrival_calendar)
            prev_sojourn, prev_arrival, arrival_count, nis, nis_vec = \
                write_event(nis, generator, residuals, departure_calendar,
                            log_scale, generated_log, arrival_count, kmeans, gamma_parms, generator_flag, nis_vec,
                            arr_timestamp, new_class, prev_arrival, prev_sojourn, context, last_los_vec, prev_class, col_names, intervention)
            prev_class = new_class

        elif arrival_calendar[0][0] <= departure_calendar[0][0]:
            


            arr_timestamp, new_class, context = heapq.heappop(arrival_calendar)
            prev_sojourn, prev_arrival, arrival_count, nis, nis_vec = write_event(nis,  generator, residuals, departure_calendar,
                            log_scale, generated_log, arrival_count, kmeans, gamma_parms, generator_flag, nis_vec, 
                            arr_timestamp, new_class, prev_arrival, prev_sojourn, context, last_los_vec, prev_class, col_names, intervention)
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

def get_class_dfs(generated_log, y, original_df):  
    # take only patients after first 100
    for k, v in generated_log.items():
        generated_log[k] = v[100:]
            
    actual_sojourn = y.to_numpy()[100:]
    simulated_sojourn = generated_log['s_sojourn']
    
    # split dataframe into all_classes + 1 dfs
    new_d = {"sojourn": actual_sojourn, "s_sojourn": simulated_sojourn, "class": original_df["class"][100:], "arr_nis": original_df["arr_nis"][100:], "arr_nis_class_0": original_df["arr_nis_class_0"][100:], "arr_nis_class_1": original_df["arr_nis_class_1"][100:], "arr_nis_class_2": original_df["arr_nis_class_2"][100:], "s_arr_nis": generated_log["s_arr_nis"], "s_arr_nis_class_0": generated_log["s_arr_nis_class_0"], "s_arr_nis_class_1": generated_log["s_arr_nis_class_1"], "s_arr_nis_class_2": generated_log["s_arr_nis_class_2"], "arrival_count": generated_log["arrival_count"], "cum_arrival": original_df["cum_arrival"][100:], "arrival": generated_log["arrival"]}
    all_classes_df = pd.DataFrame(data=new_d)
    new_dfs = [all_classes_df]
    for c in all_classes:
        new_dfs.append(all_classes_df[all_classes_df["class"] == c])

    return new_dfs
    
def get_features(specification):
    # returns the features needed for the simulation given a specification
    features = features = ['class', 'Triage Code', 'Age Category', 'Initial Zone', 'Gender Code', 'Ambulance', 'Consult', 'Admission']   
    match specification:
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
    return features
        
def run_all_generators(main_df, class_types, gen_names, class_names, spec_names, time_name, runs, prev_gens = None, prev_kmeans_gamma = None, prev_gen_flag = False, log_scale = False, generator_flag = True):
    '''
    method runs all generators and all specifications and graphs qq plots, box whisker plots, and time series graphs
    used for initial training and testing to determine the best generator
    if run on training - returns generators to use
    if run on testing - returns best generator 
    '''
    final_generators = {}
    kmeans_gamma = {}
    ks_stats = []
    stat_nis_dict = {}
    stat_soj_dict = {}
    for c in class_names:
        ks_stats.append({"vals": ["p_nis", "p_sojourn", "stat_nis", "stat_sojourn"]})
    save_path = os.path.join(os.getcwd(), "ks_stats.xlsx")
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter') 
    workbook=writer.book
    classes = main_df['class'].copy(deep=True).astype(int).to_numpy()
    best_gen = []
    
    
    for nis_residual_spec in range(4):
        spec_name = spec_names[nis_residual_spec]
        features = get_features(nis_residual_spec)
        arrivals = get_arrivals(main_df)
        context = main_df[features]
        context = pd.get_dummies(data=context, drop_first=True)
        col_names = context.columns.tolist()
        print(col_names, len(context.columns))
        # saves generators for later use
        if prev_gen_flag:
            generators, residuals = prev_gens[spec_name]
            kmeans, gamma_params = prev_kmeans_gamma[spec_name]
        else:
            generators, residuals = get_generators(context, main_df,  class_types, classes)
            K_max = 20
            k_best = find_best_k(context, main_df["sojourn"], range(2,K_max))
            kmeans, gamma_params = fit_gamma(main_df, context, y_label='sojourn', K=k_best)
            kmeans_gamma[spec_name] = (kmeans, gamma_params)
        final_generators[spec_name] = (generators, residuals)
        kmeans_gamma[spec_name] = (kmeans, gamma_params)
        
        class_spec_dfs = []
        for r in range(runs):    
            for i in range(len(gen_names)):
                generator_flag = True
                gen_name = gen_names[i]
                print(spec_name + ", " + gen_name)

                if generators[i] == False and residuals[i] == False:
                    generator_flag = False # clustering

                start = timer()
                generated_log = simulate_system(generators[i], residuals[i], log_scale,
                                            arrivals, context,  kmeans, gamma_params, generator_flag, class_types, classes, col_names, intervention = 0)
                
                end = timer()
                print("time for run:", end-start)
                #write results
                pd.DataFrame.from_dict(generated_log).to_csv(time_name +" " +spec_name + '_results_'+ str(gen_names[i]) + "_" +str(r)+'.csv', index=False)
                print("saved results\n")

                class_dfs = get_class_dfs(generated_log, main_df["sojourn"], main_df)
                # statistics gathered to determine best generator
                p_nis, p_sojourn, stat_nis, stat_sojourn = get_ks_test_vals(class_dfs, class_names)
                for c in range(len(class_names)):
                    final_stats = [p_nis[c], p_sojourn[c], stat_nis[c], stat_sojourn[c]]
                    ks_stats[c][gen_name] = final_stats
                    if c == 0:
                        stat_nis_dict[spec_name + "-" + gen_name] = stat_nis[c]
                        stat_soj_dict[spec_name + "-" + gen_name] = stat_sojourn[c]
            
                graph_hist_cumulative_ts(class_dfs, gen_name, class_names, spec_name, time_name)
                class_spec_dfs.append(class_dfs)
                
            plot_nis_ts(class_spec_dfs, gen_names, class_names, spec_name, time_name)
            plot_graphs(class_spec_dfs, gen_names, class_names, spec_name, time_name)
        
        worksheet=workbook.add_worksheet(spec_name)
        writer.sheets[spec_name] = worksheet
        for ind, name in enumerate(class_names):
            ks_df = pd.DataFrame.from_dict(ks_stats[ind])
            row = ind*7+1
            worksheet.write(row-1, 0, name)
            # write stats to excel
            ks_df.to_excel(writer, sheet_name=spec_name, startrow=row, index=False)

    if prev_gen_flag:
        sorted_nis = sorted(stat_nis_dict.items(), key=lambda x:x[1])
        sorted_soj = sorted(stat_soj_dict.items(), key=lambda x:x[1])
        # print("sorted_nis:", sorted_nis)
        # print("sorted_soj:", sorted_soj)
        gen_counters = {}
        for ind, (gen_spec, val) in enumerate(sorted_nis):
            gen_counters[gen_spec] = ind
        for ind, (gen_spec, val) in enumerate(sorted_soj):
            gen_counters[gen_spec] = gen_counters[gen_spec] + ind
        # print(gen_counters)
        best_gen = min(gen_counters, key=gen_counters.get).split("-")

    writer._save()
    # print(sums)
    # print(min(sums, key=sums.get))
    
    return best_gen, final_generators, kmeans_gamma

def run_interventions(main_df, gen_to_use, res_to_use, class_types, class_names, runs, interventions, time_name, nis_res_spec, log_scale = False, generator_flag = True):
    '''
    function to use for interventions
    given a list of interventions, simulates the system 
    '''
    
    classes = main_df['class'].copy(deep=True).astype(int).to_numpy()
    features = get_features(nis_res_spec)
    arrivals = get_arrivals(main_df)
    context = main_df[features]
    context = pd.get_dummies(data=context, drop_first=True)
    col_names = context.columns.tolist()
    print(col_names, len(context.columns))
    
    class_spec_dfs = {}
    for intervention in interventions:
        for r in range(runs):    
            start = timer()
            if generator_flag:
                gen = gen_to_use
                res = res_to_use
                kmeans = "no_kmeans"
                gamma_params = "no_gamma"
            else:
                gen = "no_gen"
                res = "no_res"
                kmeans = gen_to_use
                gamma_params = res_to_use
        

            generated_log = simulate_system(gen, res, log_scale,
                                        arrivals, context,  kmeans, gamma_params, generator_flag, class_types, classes, col_names, intervention)
            end = timer()
            print("time for run:", end-start)
            #write results
            filename = str(intervention) + "_results"
            pd.DataFrame.from_dict(generated_log).to_csv("{}.csv".format(filename), index=False)
            print("saved results\n")

            class_dfs = get_class_dfs(generated_log, main_df["sojourn"], main_df)

            class_spec_dfs[intervention] = class_dfs
    
    # graphs box plots for each class and intervention
    prepare_box_plots(class_spec_dfs, interventions, class_names, time_name)
    
    # need to add average LOS, 90 percentile stats
    
    return 
 
if __name__ == "__main__":
    # nis_residual_specs: 
    # 0 = arr_nis + last_rem_los
    # 1 = arr_nis per class + last_rem_los
    # 2 = arr_nis + last_rem_los per class
    # 3 = arr_nis per class + last_rem_los per class

    pre_processed = True
    if pre_processed:
        df = pd.read_pickle("FirstThree_cleaned.pkl")
    else:
        df = main_process("FirstThree.csv")
    
    # splits data frame given start and end dates
    training = ["1/1/2016", "2/1/2016"]
    last_month = datetime.strptime(training[1], '%m/%d/%Y') - timedelta(days=1)
    last_month = last_month.strftime("%m/%d/%Y")
    training_months = [datetime.strptime(date, '%m/%d/%Y').strftime('%b') for date in [training[0], last_month]]
    testing = ["2/1/2016", "3/1/2016"]
    last_month = datetime.strptime(testing[1], '%m/%d/%Y') - timedelta(days=1)
    last_month = last_month.strftime("%m/%d/%Y")
    testing_months = [datetime.strptime(date, '%m/%d/%Y').strftime('%b') for date in [testing[0], last_month]]
    print(training_months)
    print(testing_months)
    train_df = df[df["Triage DateTime"] >= training[0]]
    train_df = train_df[train_df["Triage DateTime"] < training[1]]
    test_df = df[df["Triage DateTime"] >= testing[0]]
    test_df = test_df[test_df["Triage DateTime"] < testing[1]]
    print("dataframes loaded")

    # sets up lists for future use
    all_classes = df['class'].astype(int).unique().tolist()
    all_classes.sort() 
    generator_names = ["EBM", "RF", "Clustering"]
    class_names = ["All Classes", "Class 0", "Class 1", "Class 2"]
    specification_names = ["1NIS_1Res", "3NIS_1Res", "1NIS_3Res", "3NIS_3Res"]

    # creates generators and simulates system on training df
    _, trained_gens, trained_clustering = run_all_generators(train_df, all_classes, generator_names, class_names, specification_names, "Train(" + training_months[0] + "-" + training_months[1] + ")", 1)
    print("done training")
    # uses the generators from training to simulate system and determine best generator for future use
    best_gen, gens, clustering = run_all_generators(test_df, all_classes, generator_names, class_names, specification_names, "Test(" + testing_months[0] + ")", 1, trained_gens, trained_clustering, prev_gen_flag=True)
    print("best generator", best_gen)
    
    # set up variables for future use
    best_gen_spec = best_gen[0]
    spec_num = specification_names.index(best_gen_spec)
    if best_gen[1] != "Clustering":
        spec_gens, spec_res = gens[best_gen_spec]
        index = generator_names.index(best_gen[1])
        gen_to_use = spec_gens[index]
        res_to_use = spec_res[index]
        generator_flag = True
    else:
         spec_kmeans, spec_params = clustering[best_gen_spec]
         gen_to_use = spec_kmeans
         res_to_use = spec_params
         generator_flag = False    


    interventions = [0, 60, 180, 300]
    # runs interventions on other dfs
    # change from test_df
    run_interventions(test_df, gen_to_use, res_to_use, all_classes, class_names, 1, interventions, "Int:" + testing_months[0], spec_num, False, generator_flag)

