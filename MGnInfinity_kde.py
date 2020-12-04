import pandas as pd
from resource import *
import heapq as hq
import numpy as np
import os
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from openpyxl import Workbook

class multi_class_MGnInfinity:
    # defining the queueing system using given parameters
    def __init__(self, **kwargs):

        # initialize parameters
        self.lambda_ = kwargs.get('lambda_', 1)  # arrival rate
        self.classes_ = kwargs.get('classes', [0])  # class types

        self.probs = kwargs.get('probs', [1])  # probability of arrival for each class
        #pre-intervention mus
        self.mus = kwargs.get('mus',[1])  # service rate without intervention
        self.probs_speedup = kwargs.get('prob_speedup', [0]*len(self.classes_))  # probability of speedup

        # self.mus_speedup = kwargs.get('mus_speedup', self.mus)  # service rate with intervention
        self.servers = kwargs.get('servers', 1)  # number of servers

        self.laplace_params = kwargs.get('laplace_params', [0, 0.5])  # location and scale parameters

        # initialize trackers with relevant statistics, assume all start empty
        self.data = []  # event logs
        self.los_tracker = []  # [[[timestamp, timestamp, ...], ...]]
        self.nis_tracker = []  # [[[(timestamp, NIS), (timestamp, NIS), ...], ...]]

    # Getters
    def get_classes(self):
        return self.classes_

    def get_los_tracker(self):
        return self.los_tracker

    def get_nis_tracker(self):
        return self.nis_tracker

    def get_event_logs(self):
        return self.data


    def simulate_MGnInfinity(self, customers, runs, given_data_path, method=1, n_clusters=0):

        np.random.seed(3)  # set random seed

        if method == 1:
            self.mus = self.method1_linear_regression(filepath=given_data_path)
        elif method == 2:
            _, kde_elapsed_method2 = self.method2_one_kde_for_all_q(filepath=given_data_path)
        elif method == 3:
            kmeans_model_method3, cluster_to_kde_method3 = self.method3_kmeans_clustering(filepath=given_data_path, nClusters=n_clusters)

        for r in range(runs):
            print('running simulation run #: ' + str(r + 1))

            t_ = 0  # time starts from zero
            sim_arrival_times = []
            classes_ = []
            interv_ = []

            for c in range(customers):
                # simulate next arrival
                # next arrival time: t_ + inter-arrival_time
                sim_arrival_times.append(
                    t_ + Distribution(dist_type=DistributionType.exponential, rate=self.lambda_).sample())

                # move forward the timestamp
                t_ = sim_arrival_times[len(sim_arrival_times)-1]

                # sampling the class of arriving patient
                c_ = np.random.choice(self.classes_, p=self.probs)
                classes_.append(c_)

                # sampling whether intervention or not
                interv_.append(np.random.choice(np.array([0, 1]),
                                                p=np.array([1 - self.probs_speedup[c_], self.probs_speedup[c_]])))


            curr_nis = 0  # current number of customers in the system
            event_log = []
            nis_tr = [[(0, 0)] for _ in self.classes_]  # [[(timestamp, NIS), (timestamp, NIS), ...], ...]
            los_tr = [[] for _ in self.classes_]  # [[timestamp, timestamp, ...], ...]
            # two types of events: arrival = 'a', departure = 'd'
            # every tuple is (timestamp, event_type, customer id, server_id)
            event_calendar = [(a, 'a', i) for i,a in enumerate(sim_arrival_times)]
            hq.heapify(event_calendar)

            # keep going if there are still events waiting to occur
            while len(list(event_calendar))>0:
                # take an event from the event_calendar
                ts_, event_, id_ = hq.heappop(event_calendar)

                # arrival event happens, need to check if servers are available
                if event_ == 'a':
                    curr_nis += 1  # update current number of customers in the system

                    # log arrival event
                    event_log.append((ts_, 'a', id_, interv_[id_], classes_[id_], curr_nis))

                    # update nis_tracker - add 1 to the class in which the customer belongs to
                    nis_tr[classes_[id_]].append((ts_, nis_tr[classes_[id_]][-1][1] + 1))

                    # update nis_tracker for all other classes, NIS stays the same
                    for class_ in self.classes_:
                        if classes_[id_] != class_:
                            nis_tr[class_].append((ts_, nis_tr[class_][-1][1]))

                    c_customer = classes_[id_]
                    #  as the first step, there is always no intervention
                    #  NEW HERE! Service time distribution is determined by 1 of the 3 methods
                    if method == 1:
                        service_time = Distribution(dist_type=DistributionType.exponential,
                                                    rate=self.mus[c_customer]/curr_nis).sample()
                    elif method == 2:
                        service_time = kde_elapsed_method2.sample(n_samples=1)[0][0]
                        while service_time < 0:
                            service_time = kde_elapsed_method2.sample(n_samples=1)[0][0]
                    elif method == 3:
                        cluster_number = kmeans_model_method3.predict(np.int64(curr_nis).reshape(-1, 1))
                        kde_for_cluster = cluster_to_kde_method3.get(cluster_number[0])
                        service_time = kde_for_cluster.sample(n_samples=1)[0][0]
                        while service_time < 0:
                            service_time = kde_for_cluster.sample(n_samples=1)[0][0]

                    # if interv_[id_] == 0:
                    #     service_time = Distribution(dist_type=DistributionType.exponential,
                    #                                 rate=self.mus[c_customer]/curr_nis).sample()
                    # else:
                    #     service_time = Distribution(dist_type=DistributionType.exponential,
                    #                                 rate=self.mus_speedup[c_customer]/curr_nis).sample()

                    # add a departure event to the event_calendar
                    hq.heappush(event_calendar, (ts_ + service_time, 'd', id_))

                    # log departure events
                    event_log.append((ts_ + service_time, 'd', id_, interv_[id_], classes_[id_], curr_nis))

                    # update los_tracker, los = current time + service time - arrival time
                    los_tr[classes_[id_]].append(ts_ + service_time - sim_arrival_times[id_])

                # departure event happens
                else:
                    curr_nis -= 1  # update current number of customers in the system
                    # update nis_tracker - subtract 1 to the class in which the customer belongs to
                    nis_tr[classes_[id_]].append((ts_, nis_tr[classes_[id_]][-1][1] - 1))

                    # update nis_tracker for all other classes, NIS stays the same
                    for class_ in self.classes_:
                        if classes_[id_] != class_:
                            nis_tr[class_].append((ts_, nis_tr[class_][-1][1]))


            # add the event_log to "data", and append trackers for each run to overall trackers
            self.data.append(event_log)
            self.los_tracker.append(los_tr)
            self.nis_tracker.append(nis_tr)

        print('Done simulating...')


    def method1_linear_regression(self, filepath):
        df = pd.read_csv(filepath)
        data = df[['NIS', 'elapsed']].reset_index()
        data_nis = data[['NIS']].to_numpy()
        data_elapsed = data[['elapsed']].to_numpy()
        model = LinearRegression(fit_intercept=True)
        model.fit(np.array(data_nis).reshape(-1, 1), np.array(data_elapsed))

        print("Model slope:       ", model.coef_[0])
        print("Model intercept:   ", model.intercept_)
        print("mu = 1/Model slope:", 1 / model.coef_[0])
        return 1 / model.coef_[0]


    # fit 1 kde to all NIS values
    def method2_one_kde_for_all_q(self, filepath):
        df = pd.read_csv(filepath)
        data = df[['NIS', 'elapsed']].reset_index()

        # KDE for Q / number in system
        data_nis = data[['NIS']].to_numpy()
        kde_nis = KernelDensity(kernel='gaussian').fit(data_nis)
        # samples_nis = kde_nis.sample(n_samples=30000)

        # KDE for elapsed / length of stay
        data_elapsed = data[['elapsed']].to_numpy()
        kde_elapsed = KernelDensity(kernel='gaussian').fit(data_elapsed)

        return kde_nis, kde_elapsed


    # clustering
    def method3_kmeans_clustering(self, filepath, nClusters):
        df = pd.read_csv(filepath)
        data = df[['NIS', 'elapsed']].reset_index()
        data_all_q = data['NIS'].to_numpy()

        # data_points = []
        # for i, row in data.iterrows():
        #     data_points.append([row[1], row[2]])  # 2 dimensional [x, y], x = NIS, y = length of stay

        kmeans_model = KMeans(n_clusters=nClusters)
        # print(data_q)
        kmeans_model.fit(data_all_q.reshape(-1, 1))  # fit 1 feature q
        label_set = set(kmeans_model.labels_)
        data_list = [[] for _ in range(len(label_set))]
        for i, label in enumerate(kmeans_model.labels_):
            data_list[label].append(data_all_q[i])

        cluster_to_q_list_dict = {}  # dictionary #1 key = cluster_X, value = list of q's for the cluster
        cluster_to_kde = {}  # dictionary #2 key = cluster_X, value = kde for the cluster
        for i, q_list in enumerate(data_list):
            cluster_name = "Cluster_{}".format(i)
            q_vals = list(set(q_list))
            cluster_to_q_list_dict[
                cluster_name] = q_vals  # dictionary #1 key = cluster_X, value = list of q's for the cluster
            data_cluster = pd.DataFrame()

            for q in q_vals:
                data_cluster = data_cluster.append(data[data['NIS'] == q])

            data_cluster = data_cluster[['elapsed']].to_numpy()
            # print(q_vals, set(data_cluster['NIS'].to_numpy()))
            kde = KernelDensity(kernel='gaussian').fit(data_cluster)
            cluster_to_kde[i] = kde  # dictionary #2  key = cluster_number, value = kde for the cluster

        return kmeans_model, cluster_to_kde


    def generate_data(self, **kwargs):
        # generating data for intervention experiments
        write_file = kwargs.get('write_file', True)

        offset = 0.0  # time at the end of last run
        directory, folder = "", ""

        # iterate through each event log (simulation run) in simulation data
        for j,e_l in enumerate(self.data):
            # print("Run #"+str(j+1))

            # creating a data-frame to manage the event logs
            # one per simulation run - we will later want to compare interventions
            df = pd.DataFrame(e_l, columns=['timestamp', 'event_type', 'id', 'A', 'C', 'NIS'])
            # two things: we both want plots to see if the simulator makes sense, and create synthetic data
            # order by id and timestamp
            df.sort_values(by=['id','timestamp'], inplace=True)
            df.reset_index(drop=True,inplace=True)

            # add additional columns to the DataFrame
            df['elapsed'] = 0.0  # time elapsed since customer's arrival
            df['arrival_time'] = 0.0
            df['id_run'] = ""
            cur_id = df.at[0,'id']
            cur_start = df.at[0,'timestamp']

            # go through each event in the DataFrame
            for i in range(len(df)):
                df.at[i,'id_run'] = str(df.at[i,'id'])+"_"+str(j)

                # if the event corresponds to the current customer
                if cur_id == df.at[i,'id']:
                    df.at[i, 'arrival_time'] = cur_start + offset
                    df.at[i,'elapsed'] = df.at[i,'timestamp'] - cur_start
                    #print(df.at[i,'event_type'])
                    #input("Press Enter to continue...")

                # if the event does not correspond to the current customer, events for the next customer starts
                else:
                    cur_id = df.at[i, 'id']  # set current customer to the customer for the event
                    cur_start = df.at[i,'timestamp'].copy()  # advance the current start time to the time of event
                    df.at[i,'arrival_time'] = cur_start + offset


            offset = offset + max(df['timestamp']) # the next simulation run starts at the offset time

            # generate csv files
            if write_file:
                # save generated data in a folder in the current working directory
                cwd = os.getcwd() # get current working directory
                # single type of customers
                if len(self.mus) == 1:
                    folder = "MG(n)Infinity - Lambda{}Mu{}P(Interv){}MuPrime{}".format(self.lambda_, self.mus[0],
                                                                                self.probs_speedup[0],
                                                                                self.mus_speedup[0])
                # two types of customers
                elif len(self.mus) == 2:
                    folder = "MG(n)Infinity - LamOne{}LamTwo{}MuOne{}MuTwo{}P(C1){}P(C1_Interv){}P(C2_Interv){}MuOnePrime{}MuTwoPrime{}".format(self.lambda_,
                                                                                                                                       self.lambda_,
                                                                                                                                       self.mus[0],
                                                                                                                                       self.mus[1],
                                                                                                                                       self.probs[0],
                                                                                                                                       self.probs_speedup[0],
                                                                                                                                       self.probs_speedup[1],
                                                                                                                                       self.mus_speedup[0],
                                                                                                                                       self.mus_speedup[1])
                # todo: more than 2 types of customers?

                directory = os.path.join(cwd, folder)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # generate file: 1) Queue/Waiting and System/Time

                header_ = True if j == 0 else False
                mode_ = 'w' if j == 0 else 'a'
                # file_1: System/Time
                filename = "data_TIS"
                save_path = os.path.join(directory, filename+".csv")
                df[df.event_type == 'd'].loc[:,
                ['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'NIS', 'elapsed']].to_csv(
                    save_path, mode=mode_, index=False, header=header_)


        # generate files: 2) System/Number
        if write_file:
            # file_2: System/Number
            filename = "data_NIS"
            save_path = os.path.join(directory, filename + ".csv")
            for r, nis_tr in enumerate(self.nis_tracker):
                df_nis = pd.DataFrame(columns=['run', 'timestamp', 'class_id', 'Number_in_System'])
                offset = 0
                for class_ in self.classes_:
                    for i, system in enumerate(nis_tr[class_]):
                        df_nis.loc[i + offset] = [r+1, system[0], class_, system[1]]
                        offset += len(nis_tr[class_])

                df_nis.sort_values(by=['timestamp'], inplace=True)  # order by timestamp
                df_nis.reset_index(drop=True, inplace=True)

                if r == 0:
                    df_nis.to_csv(save_path,index=False,header=True)
                else:
                    df_nis.to_csv(save_path, mode='a', index=False, header=False)


#  calculate the true mean & stdev from given data
def true_mean_stdev(filepath):
    df = pd.read_csv(filepath)
    data_nis = df[['NIS']].to_numpy()
    data_elapsed = df[['elapsed']].to_numpy()
    return np.mean(data_nis), np.std(data_nis), np.mean(data_elapsed), np.std(data_elapsed)


def compute_performance_measure(queueing_system, confidence_level, nReplications):

    # Number in System
    nis_num_all_classes, nis_num_all_classes_stdev, nis_num_all_classes_ci = nis(queueing_system, confidence_level, nReplications)

    # Time in System
    tis_mean_all, tis_mean_all_stdev, tis_mean_all_ci = tis(queueing_system, confidence_level, nReplications)

    # ------------------------------------------- PRINTING RESULTS -------------------------------------------
    print("Single Class FCFS M/G(n)/Infinity Numerical Results")
    return nis_num_all_classes, nis_num_all_classes_stdev, nis_num_all_classes_ci, tis_mean_all, tis_mean_all_stdev, tis_mean_all_ci


def tis(queueing_system, confidence_level, nReplications):

    tracker = queueing_system.get_los_tracker()
    mean_time_all_classes, stdev_time_all_classes = [], []
    for rep in tracker:
        # Expected Values
        time_list_all_classes = []
        for time_list in rep:
            time_list_all_classes = np.concatenate([time_list_all_classes, time_list])

        mean_time_all_classes.append(np.mean(time_list_all_classes))
        stdev_time_all_classes.append(np.std(time_list_all_classes))

    # All Classes - Mean Time
    avg_mean_tis_all_classes, avg_stdev_tis_all_classes, ci_tis_all_classes = compute_mean_stdev_ci(
        mean_time_all_classes, stdev_time_all_classes, confidence_level, nReplications)

    return avg_mean_tis_all_classes, avg_stdev_tis_all_classes, ci_tis_all_classes


def nis(queueing_system, confidence_level, nReplications):
    classes = queueing_system.get_classes()
    tracker = queueing_system.get_nis_tracker()
    mean_numbers_all_classes, stdev_numbers_all_classes = [], []

    for rep in tracker:
        t_n = rep[0][-1][0]
        area_all_classes = 0
        num_list_all_classes = []  # [n1, n2, n3, ...] N's are not unique

        for c in classes:
            num_list_all_classes += [tup[1] for tup in rep[c]]  # list of N's for customers of all classes
        unique_num_all_classes = sorted(set(num_list_all_classes))  # unique list of N's for all classes
        num_pmf_all_classes = np.zeros(len(unique_num_all_classes))

        for c in classes:
            class_x_tracker = rep[c]
            area_by_class = 0

            for i in range(1, len(class_x_tracker)):
                t_i1, t_i0 = class_x_tracker[i][0], class_x_tracker[i - 1][0]
                Num_i0 = class_x_tracker[i - 1][1]

                # E(NIQ) = sum_i=1_to_n_ [(t_i - t_i-1)*NIQ_i] / t_n; E(NIS) = sum_i=1_to_n_ [(t_i - t_i-1)*NIS_i] / t_n
                area_by_class += (t_i1 - t_i0) * Num_i0

                # For pmf of all classes
                idx = unique_num_all_classes[Num_i0]
                num_pmf_all_classes[idx] += t_i1 - t_i0

            # For expected values
            area_all_classes += area_by_class

        # Expected Value
        E_X = area_all_classes / t_n
        mean_numbers_all_classes.append(E_X)  # [rep1_all_mean, rep2_all_mean, ...]

        num_pmf_all_classes = num_pmf_all_classes / (t_n * len(classes))  # normalize pmf for customers in all classes
        # Calculate the variance from pmf --> VAR(X) = E(X^2) - [E(X)]^2 = sum(X^2*p(X)) - μ^2
        # Compute E(X^2)
        E_X_sq = 0
        for i, p_X in enumerate(num_pmf_all_classes):
            X = unique_num_all_classes[i]
            E_X_sq += X * X * p_X
        var_nis_all_classes = E_X_sq - E_X * E_X
        stdev_nis_all_classes = np.sqrt(var_nis_all_classes)
        stdev_numbers_all_classes.append(stdev_nis_all_classes)

    # All Classes - Mean Number
    avg_mean_nis_all_classes, avg_stdev_nis_all_classes, ci_nis_all_classes = compute_mean_stdev_ci(
        mean_numbers_all_classes, stdev_numbers_all_classes, confidence_level, nReplications)

    return avg_mean_nis_all_classes, avg_stdev_nis_all_classes, ci_nis_all_classes


def compute_mean_stdev_ci(mean_data, stdev_data, confidence_level, n):
        z_vals_dict = {0.8 : 1.28, 0.9 : 1.645, 0.95 : 1.96, 0.98 : 2.33, 0.99 : 2.58}  # confidence level : z*-value
        z_val = z_vals_dict.get(confidence_level)

        avg_mean = np.mean(mean_data)
        avg_stdev = np.mean(stdev_data)
        data_ci = (avg_mean - z_val*avg_stdev/np.sqrt(n), avg_mean + z_val*avg_stdev/np.sqrt(n))
        return avg_mean, avg_stdev, data_ci

def plot_percent_to_mean_real_data(data_path, plot_name, save_path):
    plt.figure()
    plt.title(plot_name)

    df_data = pd.read_csv(data_path)
    data_arrival_time = df_data[['arrival_time']].to_numpy()
    data_nis = df_data[['NIS']].to_numpy()
    data_tis = df_data[['elapsed']].to_numpy()

    data_percent_to_mean_nis = data_nis / np.mean(data_nis) * 100
    data_percent_to_mean_tis = data_tis / np.mean(data_tis) * 100

    plt.plot(data_arrival_time, data_percent_to_mean_nis, color='b', label='Percent to Mean NIS', linestyle='-')
    plt.plot(data_arrival_time, data_percent_to_mean_tis, color='g', label='Percent to Mean TIS', linestyle='-')

    plt.xlabel('Arrival Time')
    plt.ylabel('Percent to Mean')
    plt.legend()

    plt.savefig(save_path, dpi=600)


def plot_percent_to_mean_simulated_data(qs, true_mean_nis, true_mean_tis, plot_name, save_path):
    plt.figure()
    plt.title(plot_name)

    # Simulation - only plot the first replication
    data_arrival_time, data_nis, data_tis = [], [], []
    for i, rep in enumerate(qs.get_event_logs()):
        if i == 0:
            # one customer type
            for tup in rep:
                if tup[1] == 'a':
                    data_arrival_time.append(tup[0])
                    data_nis.append(tup[5])

    for i, rep in enumerate(qs.get_los_tracker()):
        if i == 0:
            # one customer type
            data, tis = rep[0], []
            for time_elapsed in data:
                tis.append(time_elapsed)
            data_tis.append(tis)


    data_percent_to_mean_nis = data_nis / true_mean_nis * 100
    data_percent_to_mean_tis = data_tis / true_mean_tis * 100

    plt.plot(data_arrival_time, data_percent_to_mean_nis, color='b', label='Percent to Mean NIS', linestyle='-')
    plt.plot(data_arrival_time, data_percent_to_mean_tis[0], color='g', label='Percent to Mean TIS', linestyle='-')

    plt.xlabel('Arrival Time')
    plt.ylabel('Percent to Mean')
    plt.legend()

    plt.savefig(save_path, dpi=600)


def grid_runs(mu_list, clusters_list, nCustomers=5000, nRuns=30):
    wb = Workbook()  # create a new workbook
    for mu in mu_list:
        sheet_name = "Mu={}".format(mu)
        wb.create_sheet(sheet_name)  # create a new worksheet
        ws = wb[sheet_name]  # worksheet
        df_results = pd.DataFrame(
            columns=["Data", "Mean(Q)", "Stdev(Q)", "95%CI(Q)", "Mean(elapsed)", "Stdev(elapsed)", "95%CI(elapsed)"])
        ws.append(df_results.columns.to_list())

        # Real Data
        path_nis = os.path.join(os.getcwd(),
                                "grid_run_data/MG(n)Infinity - Lambda1Mu{}P(Interv)0.0MuPrime2.5/data_NIS.csv".format(mu))
        path_tis = os.path.join(os.getcwd(),
                                "grid_run_data/MG(n)Infinity - Lambda1Mu{}P(Interv)0.0MuPrime2.5/data_TIS.csv".format(mu))
        print("\"Real Data\": method0...")
        mean_Q, mean_Q_stdev, mean_elapsed, mean_elapsed_stdev = true_mean_stdev(
            filepath=path_tis)  # mean_Q, stdev_Q, mean_elapsed, stdev_elapsed
        df_results.loc[0] = "Real Data", mean_Q, mean_Q_stdev, np.nan, mean_elapsed, mean_elapsed_stdev, np.nan
        plot_percent_to_mean_real_data(data_path=path_tis, plot_name="Real Data - {} Customers".format(nCustomers),
                                       save_path=os.path.join(os.getcwd(), 'grid_run_data/Real Data_Mu={}.png'.format(mu)))

        # Linear Regression
        print("Fit 1: fit linear regression to determine mu, exponetial distribution...")
        q_1 = multi_class_MGnInfinity(lambda_=1, classes=[0], probs=[1.0], mus=[1.1], prob_speedup=[0.0],
                                      mus_speedup=[2.5],
                                      servers=1)
        q_1.simulate_MGnInfinity(customers=nCustomers, runs=nRuns, given_data_path=path_tis, method=1)
        mean_Q_method1, mean_Q_stdev_method1, _, mean_elapsed_method1, mean_elapsed_stdev_method1, _ = compute_performance_measure(
            queueing_system=q_1, confidence_level=0.95, nReplications=nRuns)
        # df_results.loc[1] = mean_Q_method1, mean_Q_stdev_method1, mean_Q_CI_method1, mean_elapsed_method1, mean_elapsed_stdev_method1, mean_elapsed_CI_method1
        df_results.loc[1] = "Fit 1", mean_Q_method1, mean_Q_stdev_method1, np.nan, mean_elapsed_method1, mean_elapsed_stdev_method1, np.nan
        plot_percent_to_mean_simulated_data(qs=q_1, true_mean_nis=mean_Q_method1, true_mean_tis=mean_elapsed_method1,
                                            plot_name="Fit 1: Linear Regression - {} Customers".format(nCustomers),
                                            save_path=os.path.join(os.getcwd(), 'grid_run_data/Fit1_Mu={}.png'.format(mu)))

        # Fit 1 KDE for all Qs
        print("Fit 2: fit 1 kde for all q's...")
        q_2 = multi_class_MGnInfinity(lambda_=1, classes=[0], probs=[1.0], mus=[1.1], prob_speedup=[0.0],
                                      mus_speedup=[2.5],
                                      servers=1)
        q_2.simulate_MGnInfinity(customers=nCustomers, runs=nRuns, given_data_path=path_tis, method=2)
        mean_Q_method2, mean_Q_stdev_method2, _, mean_elapsed_method2, mean_elapsed_stdev_method2, _ = compute_performance_measure(
            queueing_system=q_2, confidence_level=0.95, nReplications=nRuns)
        # df_results.loc[2] = mean_Q_method2, mean_Q_stdev_method2, mean_Q_CI_method2, mean_elapsed_method2, mean_elapsed_stdev_method2, mean_elapsed_CI_method2
        df_results.loc[2] = "Fit 2", mean_Q_method2, mean_Q_stdev_method2, np.nan, mean_elapsed_method2, mean_elapsed_stdev_method2, np.nan
        plot_percent_to_mean_simulated_data(qs=q_2, true_mean_nis=mean_Q_method2, true_mean_tis=mean_elapsed_method2,
                                            plot_name="Fit 2: fit 1 kde for all q's - {} Customers".format(nCustomers),
                                            save_path=os.path.join(os.getcwd(), 'grid_run_data/Fit2_Mu={}.png'.format(mu)))

        # Clustering
        cluster_loc = 3
        if mu == 1.1:
            clusters_list = [5, 10, 20, 30, 40, 50]
        elif mu == 1.5 or mu == 2.0:
            clusters_list = [5, 10]
        elif mu == 2.5:
            clusters_list = [5, 8]

        for n_clusters in clusters_list:
            print("Fit 3: kmeans clustering ({} Clusters)...".format(n_clusters))
            q_3 = multi_class_MGnInfinity(lambda_=1, classes=[0], probs=[1.0], mus=[1.1], prob_speedup=[0.0],
                                          mus_speedup=[2.5], servers=1)
            q_3.simulate_MGnInfinity(customers=nCustomers, runs=nRuns, given_data_path=path_tis, method=3,
                                     n_clusters=n_clusters)
            mean_Q_method3, mean_Q_stdev_method3, _, mean_elapsed_method3, mean_elapsed_stdev_method3, _ = compute_performance_measure(
                queueing_system=q_3, confidence_level=0.95, nReplications=nRuns)
            # df_results.loc[3] = mean_Q_method3, mean_Q_stdev_method3, mean_Q_CI_method3, mean_elapsed_method3, mean_elapsed_stdev_method3, mean_elapsed_CI_method3
            df_results.loc[cluster_loc] = "Fit 3 ({} Clusters)".format(n_clusters), mean_Q_method3, mean_Q_stdev_method3, np.nan, mean_elapsed_method3, mean_elapsed_stdev_method3, np.nan
            plot_percent_to_mean_simulated_data(qs=q_3, true_mean_nis=mean_Q_method3, true_mean_tis=mean_elapsed_method3,
                                                plot_name="Fit 3: ({} Clusters) - {} Customers".format(n_clusters, nCustomers),
                                                save_path=os.path.join(os.getcwd(),'grid_run_data/Fit3_Mu={}_{}Clusters.png'.format(mu, n_clusters)))
            cluster_loc += 1

        print(df_results)

        # save to worksheet
        for row in df_results.iterrows():
            ws.append(row[1].to_list())
        wb.save(os.path.join(os.getcwd(), "grid_run_data/Results.xlsx"))


if __name__ == "__main__":
    grid_runs(mu_list=[1.1, 1.5, 2.0, 2.5], clusters_list=[5, 10], nCustomers=5000, nRuns=30)
