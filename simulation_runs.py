import pandas as pd
import os
from q_intervention import *

# test using: 1000 customers, 10 runs each
def sim2run(nCustomer, nRuns):

    # Input parameters
    lam = 1
    mu_1_list = [2, 2.5, 3]
    mu_2_list = [2, 2.5, 3]
    p_1_list = [0.2, 0.4, 0.6]
    p_intervention_list = [0, 0.3, 0.6, 0.9]
    speedup_list = [0.2, 0.3, 0.4, 0.5]
    # time interval for friends not accounted for
    count = 0

    # Running simulations with varying parameter values
    for mu1 in mu_1_list:
        for mu2 in mu_2_list:
            for p1 in p_1_list:
                p2 = 1 - p1
                # Calculate the value of rho, run the simulation only if rho < 1
                rho = lam /(p1 * mu1 + p2 * mu2)
                if rho >= 1:
                    continue
                for p_interv in p_intervention_list:
                    for speedup in speedup_list:
                        mu1_sp = mu1 * (1 + speedup)
                        mu2_sp = mu2 * (1 + speedup)
                        # Assumptions:
                        # 1) 2 class types
                        # 2) 1 single server
                        # 3) prob_speedup for the second class is always zero
                        q_ = multi_class_single_station_fcfs(lambda_ = lam, classes = [0,1], probs = [p1,p2],
                                     mus = [mu1,mu2], prob_speedup=[p_interv,0.0], mus_speedup=[mu1_sp, mu2_sp],
                                     servers = 1)

                        q_.simulate_q(customers=nCustomer, runs=nRuns)

                        q_.generate_data(sla_=0.9, quant_flag=True, write_file=True)

                        cwd = os.getcwd()  # get current working directory
                        folder = "Lambda{}Mu1{}Mu2{}p1{}ProbIntervention{}Mu1Speedup{}Mu2Speedup{}".format(lam,
                        mu1, mu2, p1, p_interv, mu1_sp, mu2_sp)
                        directory = os.path.join(cwd, folder)

                        # try printing the performance measures values for the first 3 runs
                        if count < 3:
                            print('Simulation Run #', count+1)
                            print("rho: ", rho)

                            average_tis, average_tis_sd, percentiles_all_tis, percentiles_C1_tis, percentiles_C2_tis = system_time(
                                directory)
                            print("Mean TIS (All, Class 1, Class 2): ", average_tis)
                            print("TIS Standard Deviation (All, Class 1, Class 2): ", average_tis_sd)
                            print("TIS Percentiles All: ", percentiles_all_tis)
                            print("TIS Percentiles Class 1: ", percentiles_C1_tis)
                            print("TIS Percentiles Class 2: ", percentiles_C2_tis)

                            average_wiq, average_wiq_sd, percentiles_all_wiq, percentiles_C1_wiq, percentiles_C2_wiq = queue_waiting(
                                directory)
                            print("Mean WIQ (All, Class 1, Class 2): ", average_wiq)
                            print("WIQ Standard Deviation (All, Class 1, Class 2): ", average_wiq_sd)
                            print("WIQ Percentiles All: ", percentiles_all_wiq)
                            print("WIQ Percentiles Class 1: ", percentiles_C1_wiq)
                            print("WIQ Percentiles Class 2: ", percentiles_C2_wiq)

                            average_niq, average_niq_sd, percentiles_all_niq, percentiles_C1_niq, percentiles_C2_niq = queue_number(
                                directory)
                            print("Mean NIQ (All, Class 1, Class 2): ", average_niq)
                            print("NIQ Standard Deviation (All, Class 1, Class 2): ", average_niq_sd)
                            print("NIQ Percentiles All: ", percentiles_all_niq)
                            print("NIQ Percentiles Class 1: ", percentiles_C1_niq)
                            print("NIQ Percentiles Class 2: ", percentiles_C2_niq)

                            average_nis, average_nis_sd, percentiles_all_nis, percentiles_C1_nis, percentiles_C2_nis = system_number(
                                directory)
                            print("Mean NIS (All, Class 1, Class 2): ", average_nis)
                            print("NIS Standard Deviation (All, Class 1, Class 2): ", average_nis_sd)
                            print("NIS Percentiles All: ", percentiles_all_nis)
                            print("NIS Percentiles Class 1: ", percentiles_C1_nis)
                            print("NIS Percentiles Class 2: ", percentiles_C2_nis)
                            if count == 2: return
                        count+=1

def calculate_sd(data_list):
    sd_list = []
    for data in data_list:
        N = len(data)
        var = ((np.std(data))**2)*(N/(N-1))
        sd_list.append(np.sqrt(var))
    return sd_list

# helper function to calculate the average number in queue or system
def calculate_average_number(data_list, queue=True):
    total_num_list = []
    col = 'Number_in_Queue' if queue else 'Number_in_System'
    t_n = data_list[0].at[len(data_list[0]) - 1, 'timestamp']
    for data in data_list:
        total_num = 0
        for i in range(1, len(data)):
            area = (data.at[i, 'timestamp'] - data.at[i-1, 'timestamp']) * data.at[i, col]
            total_num += area
        total_num_list.append(total_num / t_n)
    return total_num_list

# calculate percentiles
def calculate_percentiles(data_all, data_C1, data_C2):
    percentile_all = [np.percentile(a=data_all, q=x) for x in range(10, 101, 10)]
    percentile_C1 = [np.percentile(a=data_C1, q=x) for x in range(10, 101, 10)]
    percentile_C2 = [np.percentile(a=data_C2, q=x) for x in range(10, 101, 10)]
    return percentile_all, percentile_C1, percentile_C2

def queue_waiting(directory):
    _, df_all_wiq, _, df_C1_wiq, _, df_C2_wiq = read_in_csv(directory, "data_WIQ_TIS")
    # Mean values and standard deviation calculations, stored as [all, class 1, class 2, ...]
    average_wiq = [np.mean(df_all_wiq['elapsed']), np.mean(df_C1_wiq['elapsed']), np.mean(df_C2_wiq['elapsed'])]
    average_wiq_sd = calculate_sd([df_all_wiq['elapsed'], df_C1_wiq['elapsed'], df_C2_wiq['elapsed']])
    # Percentiles, stored as [10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 100th]
    all_wiq, C1_wiq, C2_wiq = calculate_percentiles(df_all_wiq['elapsed'], df_C1_wiq['elapsed'], df_C2_wiq['elapsed'])
    return average_wiq, average_wiq_sd, all_wiq, C1_wiq, C2_wiq

def system_time(directory):
    df_all_tis, _, df_C1_tis, _, df_C2_tis, _ = read_in_csv(directory, "data_WIQ_TIS")
    # Mean values and standard deviation calculations, stored as [all, class 1, class 2, ...]
    average_tis = [np.mean(df_all_tis['elapsed']), np.mean(df_C1_tis['elapsed']), np.mean(df_C2_tis['elapsed'])]
    average_tis_sd = calculate_sd([df_all_tis['elapsed'], df_C1_tis['elapsed'], df_C2_tis['elapsed']])
    # Percentiles, stored as [10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 100th]
    all_tis, C1_tis, C2_tis = calculate_percentiles(df_all_tis['elapsed'], df_C1_tis['elapsed'], df_C2_tis['elapsed'])
    return average_tis, average_tis_sd, all_tis, C1_tis, C2_tis

def queue_number(directory):
    df_all_niq, df_C1_niq, df_C2_niq = read_in_csv(directory, "data_NIQ")
    # Mean values and standard deviation calculations, stored ad [all, class 1, class 2, ...]
    # E(NIQ) = sum_i=1_to_n_ [(t_i - t_i-1)*NIQ_i] / t_n
    average_niq = calculate_average_number([df_all_niq, df_C1_niq, df_C2_niq], True)
    average_niq_sd = calculate_sd([df_all_niq['Number_in_Queue'], df_C1_niq['Number_in_Queue'],df_C2_niq['Number_in_Queue']])
    # Percentiles, stored as [10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 100th]
    all_niq, C1_niq, C2_niq = calculate_percentiles(df_all_niq['Number_in_Queue'], df_C1_niq['Number_in_Queue'], df_C2_niq['Number_in_Queue'])
    return average_niq, average_niq_sd, all_niq, C1_niq, C2_niq

def system_number(directory):
    df_all_nis, df_C1_nis, df_C2_nis = read_in_csv(directory, "data_NIS")
    # Mean values and standard deviation calculations, stored ad [all, class 1, class 2, ...]
    # E(NIQ) = sum_i=1_to_n_ [(t_i - t_i-1)*NIQ_i] / t_n
    average_nis = calculate_average_number([df_all_nis, df_C1_nis, df_C2_nis], False)
    average_nis_sd = calculate_sd([df_all_nis['Number_in_System'], df_C1_nis['Number_in_System'],df_C2_nis['Number_in_System']])
    # Percentiles, stored as [10th, 20th, 30th, 40th, 50th, 60th, 70th, 80th, 90th, 100th]
    all_nis, C1_nis, C2_nis = calculate_percentiles(df_all_nis['Number_in_System'], df_C1_nis['Number_in_System'],
                                                    df_C2_nis['Number_in_System'])
    return average_nis, average_nis_sd, all_nis, C1_nis, C2_nis

def read_in_csv(directory, filename):

    if filename == "data_WIQ_TIS":
        # file_1: System/Time (TIS) and Queue/Waiting (WIQ)
        file_path_1 = os.path.join(directory, filename + ".csv")
        df_all = pd.read_csv(file_path_1, header=0)

        # condition statements
        C1, C2 = df_all.C == 0, df_all.C == 1
        TIS, WIQ = df_all.event_type == 'd', df_all.event_type == 's'

        # create DataFrames for TIS and WIQ for class 1, class 2, and altogether
        df_all_tis, df_all_wiq = df_all[TIS], df_all[WIQ]
        df_C1_tis, df_C1_wiq = df_all[C1 & TIS], df_all[C1 & WIQ]
        df_C2_tis, df_C2_wiq = df_all[C2 & TIS], df_all[C2 & WIQ]

        return df_all_tis, df_all_wiq, df_C1_tis, df_C1_wiq, df_C2_tis, df_C2_wiq

    elif filename == "data_NIQ":
        # file_2: Queue/Number (NIQ)
        file_path_2 = os.path.join(directory, filename + ".csv")
        df_all_niq = pd.read_csv(file_path_2, header=0)
        df_C1_niq, df_C2_niq = df_all_niq[df_all_niq.class_id == 0], df_all_niq[df_all_niq.class_id == 1]
        # reset indices for class 1 and class 2
        df_C1_niq = df_C1_niq.reindex(list(range(df_all_niq.index.min(),df_all_niq.index.max()+1)),fill_value=0)
        df_C1_niq = df_C1_niq.assign(timestamp=df_all_niq['timestamp'])
        df_C2_niq = df_C2_niq.reindex(list(range(df_all_niq.index.min(),df_all_niq.index.max()+1)),fill_value=0)
        df_C2_niq = df_C2_niq.assign(timestamp= df_all_niq['timestamp'])
        return df_all_niq, df_C1_niq, df_C2_niq

    elif filename == "data_NIS":
        # file_3: System/Number
        file_path_3 = os.path.join(directory, filename + ".csv")
        df_all_nis = pd.read_csv(file_path_3, header=0)
        df_C1_nis, df_C2_nis = df_all_nis[df_all_nis.class_id == 0], df_all_nis[df_all_nis.class_id == 1]
        # reset indices for class 1 and class 2
        df_C1_nis = df_C1_nis.reindex(list(range(df_all_nis.index.min(), df_all_nis.index.max() + 1)), fill_value=0)
        df_C1_nis = df_C1_nis.assign(timestamp=df_all_nis['timestamp'])
        df_C2_nis = df_C2_nis.reindex(list(range(df_all_nis.index.min(), df_all_nis.index.max() + 1)), fill_value=0)
        df_C2_nis = df_C2_nis.assign(timestamp=df_all_nis['timestamp'])
        return df_all_nis, df_C1_nis, df_C2_nis

    else:
        return "ERROR: filename not found"

if __name__ == "__main__":
    nCustomer, nRuns = 1000, 10
    sim2run(nCustomer, nRuns)