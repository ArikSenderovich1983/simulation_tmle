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
    ax.plot(gen_x_values, generated_series, label='Generated'+x_label)

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
    print("Statistic sojourn: "+x_label, statistic)
    print("P-value: "+x_label, p_value)
def compute_stats(generated_log, df):

    plot_hist(generated_log['nis_arrival'], df['arr_nis'], x_label="NIS", num_bin=50)
    plot_hist(generated_log['sojourn'], df['sojourn'], x_label="Sojourn", num_bin=50)

    # Define the bin intervals
    bin_intervals = np.arange(0, max(generated_log['arrival']), 0.0001)
    # Bin the time series data
    x_gen_arrivals = np.digitize(generated_log['arrival'], bin_intervals) - 1

    plot_ts(x_gen_arrivals, #df['arrival'].values[0:len(generated_log['arrival'])],
            generated_log['nis_arrival'], df['arr_nis'].values[0:len(generated_log['arrival'])], "NIS")
    plot_ts(x_gen_arrivals, #df['arrival'].values[0:len(generated_log['arrival'])],
           generated_log['arrival_count'], df['cum_arrival'].values[0:len(generated_log['arrival'])],
            "Cumulative Arrival")

    #plot_ts(generated_log['arrival'], df['arrival'].values[0:len(generated_log['arrival'])],  generated_log['nis_arrival'], df['arr_nis'].values[0:len(generated_log['arrival'])], "NIS")
    #plot_ts(generated_log['arrival'], df['arrival'].values[0:len(generated_log['arrival'])], generated_log['arrival_count'], df['cum_arrival'].values[0:len(generated_log['arrival'])], "Cumulative Arrival")

    ks_test(generated_log['sojourn'], df['sojourn'].values[0:len(generated_log['arrival'])], "Sojourn")
    ks_test(generated_log['nis_arrival'], df['arr_nis'].values[0:len(generated_log['arrival'])], "NIS")

#learning and bootstrapping
def learn_generator(X, y , log_scale=True, model_based_sampling=True, allow_neg=False):
        if log_scale:
            y = np.log(y)
        ebm = ExplainableBoostingRegressor()
        residuals  =[]
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
                kmeans, gamma_params, generator_flag, nis_vec, arr_timestamp, new_class, prev_arrival, prev_sojourn, context):

    #an arrival event
    #write the event down
    generated_log['arrival'].append(arr_timestamp)
    last_los = max(0,prev_sojourn - (arr_timestamp-prev_arrival))
    pred_features = [nis, new_class, last_los]
    pred_features.extend(context)
    for c in all_classes:
        pred_features.append(nis_vec[c])
    if generator_flag:
        gen_duration = [0]
        while gen_duration[0]<=0:
            gen_duration = return_func(log_scale)(generator.predict([pred_features]) + sample_with_replacement(residuals, num_samples=1))
    else:
        cur_params = gamma_params[kmeans.predict([pred_features])[0]]
        gen_duration = sample_from_gamma_clusters(1, cur_params[0], cur_params[1], cur_params[2])


    departure_timestamp = arr_timestamp + gen_duration
    #departure time is added to the calendar
    heapq.heappush(departure_calendar, (departure_timestamp[0], new_class, context))
    #departure and sojourn times documented
    generated_log['departure'].append(departure_timestamp[0])
    generated_log['sojourn'].append(departure_timestamp[0] - arr_timestamp)
    #nis upon arrival
    generated_log['nis_arrival'].append(nis)
    for c in all_classes:
        generated_log['arr_nis_class_'+str(c)].append(nis_vec[c])
    generated_log['arrival_count'].append(arrival_count)
    prev_sojourn = gen_duration[0]
    prev_arrival = arr_timestamp
    arrival_count += 1
    nis_vec[new_class] += 1
    nis = sum(nis_vec)

    return prev_sojourn, prev_arrival, arrival_count, nis, nis_vec
def simulate_system(generator, residuals, log_scale, sampled_arrivals, sampled_context, kmeans, gamma_parms, generator_flag, all_classes):
    # heapq.heapify(arrival_calendar)
    arrival_calendar = []
    departure_calendar = []

    cur_ts = 0
    eps_ = 0.00001
    all_ts_ties = []
    count_ties = 0
    count_for_context = 0
    for k,v in sampled_arrivals.items():

        for s in v:
            cur_ts += s
            while cur_ts in all_ts_ties:
                #tie break
                cur_ts = cur_ts + eps_
                count_ties+=1
            heapq.heappush(arrival_calendar, (cur_ts, k, sampled_context.iloc[count_for_context].values))
            all_ts_ties.append(cur_ts)
            count_for_context += 1
        cur_ts = 0

    print('Ties:', count_ties)
    nis = 0
    nis_vec = []
    generated_log = {'arrival': [], 'departure': [], 'sojourn': [], 'nis_arrival': [], 'arrival_count':[]}
    for c in all_classes:
        nis_vec.append(0)
        generated_log['arr_nis_class_'+str(c)] = []
    arrival_count = 0

    #first arrival
    arr_timestamp, new_class, context = heapq.heappop(arrival_calendar)
    prev_arrival = arr_timestamp
    prev_sojourn = 0

    prev_sojourn, prev_arrival, arrival_count, nis, nis_vec = write_event(nis, generator, residuals, departure_calendar, log_scale, generated_log,
                    arrival_count, kmeans, gamma_parms, generator_flag, nis_vec, arr_timestamp, new_class, prev_arrival, prev_sojourn, context)



    while len(arrival_calendar) > 0:
        if arrival_count % 500 == 0:
            print('arrivals # :', arrival_count)
            print('nis vec: ', nis_vec)
            print('total nis:', nis)

        if len(departure_calendar)==0:
            arr_timestamp, new_class = heapq.heappop(arrival_calendar)
            prev_sojourn, prev_arrival, arrival_count, nis, nis_vec = \
                write_event(nis, generator, residuals, departure_calendar,
                            log_scale, generated_log, arrival_count, kmeans, gamma_parms, generator_flag, nis_vec,
                            arr_timestamp, new_class, prev_arrival, prev_sojourn, context)

        elif arrival_calendar[0] <= departure_calendar[0]:

                arr_timestamp, new_class, context = heapq.heappop(arrival_calendar)
                prev_sojourn, prev_arrival, arrival_count, nis, nis_vec =\
                    write_event(nis,  generator, residuals, departure_calendar,
                                log_scale, generated_log, arrival_count, kmeans, gamma_parms, generator_flag, nis_vec, arr_timestamp, new_class, prev_arrival, prev_sojourn, context)

        else:
            _, new_class, _ = heapq.heappop(departure_calendar)
            nis_vec[new_class]-=1
            nis = sum(nis_vec)

    while len(departure_calendar) > 0:
        _, _, _ = heapq.heappop(departure_calendar)
        #nis_vec[new_class] -= 1
        #nis = sum(nis_vec)

    print('arr calendar: ', len(arrival_calendar))
    print('dep calendar: ', len(departure_calendar))

    return generated_log

if __name__ == "__main__":
    #initialize the experiment:
    n = 1000
    df = pd.read_csv('horizontal_multi_many.csv')
    #todo:
    #1. Arik to check log scale issue.
    #2. Daphne: use data arrivals for now
    #3. Daphne: one-hot encode "class"
    #3.1 Daphne: fix data type for timestamps
    #4. Run NYGH data through this pipeline
    #5. EBM - feature importance 
    static_context = []# ['x1', 'x2']
    features = ["arr_nis","class", "last_remaining_los"]
    for s in static_context:
        features.append(s)
    all_classes = df['class'].unique()
    for c in all_classes:
        features.append('arr_nis_class_'+str(c))
    X = df[features]
    y = df['sojourn']
    log_scale = False
    generator_flag=True
    generator, residuals = learn_generator(X, y, log_scale=log_scale,
                                           model_based_sampling=True, allow_neg=True)
    K_max = 20
    k_best = find_best_k(X, y, range(2,K_max))
    kmeans, gamma_params = fit_gamma(df, X, y_label='sojourn', K=k_best)
    # bootstrap n new arrivals (new horizon)
    runs = 3
    for r in range(runs):
        #sampled_arrivals = df['arrival'].diff().copy()#copy()#sample_arrivals(df, n)
        #sampled_arrivals = sampled_arrivals.drop(0)
        #sampled_arrivals = sampled_arrivals.values[0:n]
        sampled_arrivals, sampled_context = sample_arrivals(df, static_context,  n)
        #simulate the qnet
        generated_log = simulate_system(generator, residuals, log_scale,
                                        sampled_arrivals, sampled_context,  kmeans, gamma_params, generator_flag, all_classes)
        #compute and plot stats
        compute_stats(generated_log, df)
    #write results
        pd.DataFrame.from_dict(generated_log).to_csv('results_'+str(r)+'.csv')

