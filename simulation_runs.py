import pandas as pd
import os
from q_intervention import *

# test using: 100 customers, 10 runs each
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

                        # todo: calculate performance measures based on the simulation data
                        # todo: TEST
                        # df_all_wiq, df_C1_wiq, df_C2_wiq = queue_waiting(directory)  # elapsed column is all zeros?
                        # df_all_tis, df_C1_tis, df_C2_tis = system_time(directory)
                        if count == 0:
                            print("rho: ", rho)
                            average_tis, average_tis_sd = system_time(directory)
                            print("Mean TIS Value (All, Class 1, Class 2): ", average_tis)
                            print("TIS Standard Deviation (All, Class 1, Class 2): ", average_tis_sd)
                            return

def calculate_sd(data):
    N = len(data)
    var = ((np.std(data))**2)*(N/(N-1))
    return np.sqrt(var)

def performance_measures(folder):
    # queue_waiting
    # system_time
    # queue_number
    # system_number
    pass

def queue_waiting(directory):
    _, df_all_wiq, _, df_C1_wiq, _, df_C2_wiq = read_in_csv(directory, "data_WIQ_TIS")
    # todo: investigate why the elapsed column values are zeros
    # Mean values and standard deviation calculations
    return df_all_wiq, df_C1_wiq, df_C2_wiq

def system_time(directory):
    df_all_tis, _, df_C1_tis, _, df_C2_tis, _ = read_in_csv(directory, "data_WIQ_TIS")
    # Mean values and standard deviation calculations, stored ad [all, class 1, class 2, ...]
    average_tis = [np.mean(df_all_tis['elapsed']), np.mean(df_C1_tis['elapsed']), np.mean(df_C2_tis['elapsed'])]
    average_tis_sd = [calculate_sd(df_all_tis['elapsed']), calculate_sd(df_C1_tis['elapsed']), calculate_sd(df_C2_tis['elapsed'])]
    return average_tis, average_tis_sd

def queue_number(directory):
    df_all_niq, df_C1_niq, df_C2_niq = read_in_csv(directory, "data_NIQ")
    return

def system_number(directory):
    df_all_nis, df_C1_nis, df_C2_nis = read_in_csv(directory, "data_NIS")
    return

def read_in_csv(directory, filename):

    if filename == "data_WIQ_TIS":
        # file_1: System/Time (TIS) and Queue/Waiting (WIQ)
        file_path_1 = os.path.join(directory, filename + ".csv")
        df_all = pd.read_csv(file_path_1, header=0)

        # condition statements
        C1, C2 = df_all.C == 0, df_all.C == 1
        TIS, WIQ = df_all.event_type == 'd', df_all.event_type == 'q'

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

        return df_all_niq, df_C1_niq, df_C2_niq

    elif filename == "data_NIS":
        # file_3: System/Number
        file_path_3 = os.path.join(directory, filename + ".csv")
        df_all_nis = pd.read_csv(file_path_3, header=0)
        df_C1_nis, df_C2_nis = df_all_nis[df_all_nis.class_id == 0], df_all_nis[df_all_nis.class_id == 1]

        return df_all_nis, df_C1_nis, df_C2_nis

    else:
        return

if __name__ == "__main__":
    nCustomer, nRuns = 1000, 10
    sim2run(nCustomer, nRuns)