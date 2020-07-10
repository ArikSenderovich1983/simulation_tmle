from q_intervention import *
from scipy.stats import t, sem
import matplotlib.pyplot as plt

def simulation(nCustomer, nReplications):
    """
    Assumptions:
    1) one type of customer
    2) fixed arrival rate: lambda = 1
    3) fixed service rate before intervention: mu = 1.1
    4) 4 types of performance measures: Wait Time in Queue, Time in System, Number in Queue, Number in System
    """
    # Input parameters
    lam = 1  # fixed arrival rate
    mu = 1.1  # fixed original service rate
    mu_prime_list = np.array([2, 2.5, 3])
    p_intervention_list = np.linspace(0, 1, 11)
    confidence_level = 0.95
    count = 0

    # Calculate the value of rho, run the simulation only if rho < 1
    # rho = lam / mu
    # if rho >= 1:
    #     return

    # keep track of statistics, 3 values of mu, 11 values of P(intervention)
    tis_mean, wiq_mean, niq_mean, nis_mean = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_ci, wiq_ci, niq_ci, nis_ci = np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f')
    tis_90per, wiq_90per, niq_90per, nis_90per = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_90per_ci, wiq_90per_ci, niq_90per_ci, nis_90per_ci = np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f')

    for i, mu_prime in enumerate(mu_prime_list):
        for j, p_interv in enumerate(p_intervention_list):
            print("Parameters: mu_prime={}, p_intervention={}".format(mu_prime, p_interv))
            q_ = multi_class_single_station_fcfs(lambda_ = lam, classes = [0], probs = [1.0],
                         mus = [mu], prob_speedup=[p_interv], mus_speedup=[mu_prime],
                         servers = 1)

            q_.simulate_q(customers=nCustomer, runs=nReplications)
            q_.generate_data(sla_=0.9, quant_flag=True, write_file=False)

            # Time in System
            los_mean_all, los_mean_all_ci, los_mean_percentile90_all, los_mean_percentile90_all_ci = los(q_,
                                                                                                         confidence_level,
                                                                                                         nReplications)
            tis_mean[i-1, j-1], tis_ci[i-1, j-1] = los_mean_all, los_mean_all_ci
            tis_90per[i-1, j-1], tis_90per_ci[i-1, j-1] = los_mean_percentile90_all, los_mean_percentile90_all_ci

            # Number in Queue
            wiq_num_all_classes, wiq_num_all_classes_ci = wiq_nis(q_, confidence_level, nReplications, queue=True)
            wiq_mean[i - 1, j - 1], wiq_ci[i - 1, j - 1] = wiq_num_all_classes, wiq_num_all_classes_ci

            # Number in System
            nis_num_all_classes, nis_num_all_classes_ci = wiq_nis(q_, confidence_level, nReplications, queue=False)
            nis_mean[i - 1, j - 1], nis_ci[i - 1, j - 1] = nis_num_all_classes, nis_num_all_classes_ci

    # plot_mean_90percentile_with_CI(tis_mean, tis_ci, tis_90per, tis_90per_ci,
    #                                "Time in System With 95% Confidence Interval, {} Customers, {} Replications".format(
    #                                    nCustomer, nReplications))
    plot_mean_90percentile_with_CI(wiq_mean, wiq_ci, None, None,
                                   "Number in Queue With 95% Confidence Interval, {} Customers, {} Replications".format(
                                       nCustomer, nReplications))
    plot_mean_90percentile_with_CI(nis_mean, nis_ci, None, None,
                                   "Number in System With 95% Confidence Interval, {} Customers, {} Replications".format(
                                       nCustomer, nReplications))


def los(queueing_system, confidence_level, n):

    los_tracker = queueing_system.get_los_tracker()
    avg_los_by_class, avg_los_all_classes = [], []
    los_percentile90_by_class, los_percentile90_all_classes = [], []

    for rep in los_tracker:
        # Individual Classes
        avg_los_classes = np.mean(rep, axis=1)
        avg_los_by_class.append(avg_los_classes)
        los_percentile90_classes = np.percentile(rep, 90, axis=1)
        los_percentile90_by_class.append(los_percentile90_classes)
        # All Classes
        avg_los_all = np.mean(rep)
        avg_los_all_classes.append(avg_los_all)
        los_percentile90_all = np.percentile(rep, 90)
        los_percentile90_all_classes.append(los_percentile90_all)

    # Individual Classes
    # Expected Value
    los_mean_by_class = np.mean(avg_los_by_class, axis=0)
    # los_mean_by_class_var = ((np.std(avg_los_by_class, axis=0))**2)*(n/(n - 1))
    # los_mean_by_class_sd = np.sqrt(los_mean_by_class_var)
    los_mean_by_class_stderr = sem(avg_los_by_class, axis=0)
    los_mean_by_class_ci = t.interval(confidence_level, n - 1, los_mean_by_class, los_mean_by_class_stderr)
    # 90th Percentiles
    los_mean_percentile90_by_class = np.mean(los_percentile90_by_class, axis=0)
    los_mean_percentile90_by_class_stderr = sem(los_percentile90_by_class, axis=0)
    los_mean_percentile90_by_class_ci = t.interval(confidence_level, n - 1, los_mean_percentile90_by_class, los_mean_percentile90_by_class_stderr)

    # All Classes
    # Expected Value
    los_mean_all = np.mean(avg_los_all_classes)
    los_mean_all_stderr = sem(avg_los_all_classes)
    los_mean_all_ci = t.interval(confidence_level, n - 1, los_mean_all, los_mean_all_stderr)
    # 90th Percentile
    los_mean_percentile90_all = np.mean(los_percentile90_all_classes)
    los_mean_percentile90_all_stderr = sem(los_percentile90_all_classes)
    los_mean_percentile90_all_ci = t.interval(confidence_level, n - 1, los_mean_percentile90_all, los_mean_percentile90_all_stderr)

    return los_mean_all, los_mean_all_ci, los_mean_percentile90_all, los_mean_percentile90_all_ci


def wiq_nis(queueing_system, confidence_level, n, queue=False):
    # todo: not sure how to compute confidence interval for 90th percentile
    classes = queueing_system.get_classes()
    if queue:
        tracker = queueing_system.get_queue_tracker()
    else:
        tracker = queueing_system.get_nis_tracker()

    avg_num_list_all_classes = []
    for rep in tracker:
        t_n = rep[0][-1][0]
        avg_num_list_by_class  = []
        area_all_classes = 0
        for c in classes:
            class_x_tracker = rep[c]
            area_by_class = 0
            avg_num_list_classes = []  # [rep1_c1_mean, rep1_c2_mean, ...]


            for i in range(1, len(class_x_tracker)):
                # E(NIQ) = sum_i=1_to_n_ [(t_i - t_i-1)*NIQ_i] / t_n
                # E(NIS) = sum_i=1_to_n_ [(t_i - t_i-1)*NIS_i] / t_n
                area_by_class += (class_x_tracker[i][0] - class_x_tracker[i-1][0]) * class_x_tracker[i][1]
            avg_num_list_classes.append(area_by_class / t_n)
            area_all_classes += area_by_class

        avg_num_list_by_class.append(avg_num_list_classes)  # [[rep1_c1_mean, rep1_c2_mean, ...], [rep1_c1_mean, rep1_c2_mean, ...]]
        avg_num_list_all_classes.append(area_all_classes / t_n)  # [rep1_all_mean, rep2_all_mean, ...]

    # Individual Classes
    # Expected Values
    mean_num_by_class = np.mean(avg_num_list_by_class, axis=0)
    mean_num_by_class_stderr = sem(avg_num_list_by_class, axis=0)
    mean_num_by_class_ci = t.interval(confidence_level, n - 1, mean_num_by_class, mean_num_by_class_stderr)
    # todo: 90th Percentiles

    # All Classes
    # Expected Value
    mean_num_all_classes = np.mean(avg_num_list_all_classes)
    mean_num_all_classes_stderr = sem(avg_num_list_all_classes)
    mean_num_all_classes_ci = t.interval(confidence_level, n - 1, mean_num_all_classes, mean_num_all_classes_stderr)
    # todo: 90th Percentile

    return mean_num_all_classes, mean_num_all_classes_ci


def plot_mean_90percentile_with_CI(mean, mean_ci, percentile90, percentile90_ci, plot_type):
    x = np.linspace(0, 1, 11)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(plot_type)
    fig.set_size_inches(12, 6)

    # Plotting Expected Values
    y1, y2, y3 = mean[0], mean[1], mean[2]
    ci1_lower, ci1_upper = [mean_ci[0][i][0] for i in range(len(mean_ci[0]))], [mean_ci[0][i][1] for i in
                                                                                range(len(mean_ci[0]))]
    ci2_lower, ci2_upper = [mean_ci[1][i][0] for i in range(len(mean_ci[1]))], [mean_ci[1][i][1] for i in
                                                                                range(len(mean_ci[0]))]
    ci3_lower, ci3_upper = [mean_ci[2][i][0] for i in range(len(mean_ci[2]))], [mean_ci[2][i][1] for i in
                                                                                range(len(mean_ci[0]))]

    ax1.plot(x, y1, label='mu_prime=2')
    ax1.fill_between(x, ci1_lower, ci1_upper, color='b', alpha=0.1, label="95CI")
    ax1.plot(x, y2, label='mu_prime=2.5')
    ax1.fill_between(x, ci2_lower, ci2_upper, color='g', alpha=0.1, label="95CI")
    ax1.plot(x, y3, label='mu_prime=3')
    ax1.fill_between(x, ci3_lower, ci3_upper, color='r', alpha=0.1, label="95CI")
    ax1.set(xlabel = 'P(speedup)', ylabel = 'Expected Value')
    ax1.legend(('mu_prime=2', 'mu_prime=2.5', 'mu_prime=3'))

    # Plotting 90th Percentiles
    if percentile90 is not None and percentile90_ci is not None:
        y90p1, y90p2, y90p3 = percentile90[0], percentile90[1], percentile90[2]
        ci90p1_lower, ci90p1_upper = [percentile90_ci[0][i][0] for i in range(len(percentile90_ci[0]))], [
            percentile90_ci[0][i][1] for i in range(len(percentile90_ci[0]))]
        ci90p2_lower, ci90p2_upper = [percentile90_ci[1][i][0] for i in range(len(percentile90_ci[1]))], [
            percentile90_ci[1][i][1] for i in range(len(percentile90_ci[0]))]
        ci90p3_lower, ci90p3_upper = [percentile90_ci[2][i][0] for i in range(len(percentile90_ci[2]))], [
            percentile90_ci[2][i][1] for i in range(len(percentile90_ci[0]))]

        ax2.plot(x, y90p1, label='mu_prime=2')
        ax2.fill_between(x, ci90p1_lower, ci90p1_upper, color='b', alpha=0.1, label="95CI")
        ax2.plot(x, y90p2, label='mu_prime=2.5')
        ax2.fill_between(x, ci90p2_lower, ci90p2_upper, color='g', alpha=0.1, label="95CI")
        ax2.plot(x, y90p3, label='mu_prime=3')
        ax2.fill_between(x, ci90p3_lower, ci90p3_upper, color='r', alpha=0.1, label="95CI")
        ax2.set(xlabel='P(speedup)', ylabel='90th Percentile')
        ax2.legend(('mu_prime=2', 'mu_prime=2.5', 'mu_prime=3'))

    save_path = os.path.join(os.getcwd(), plot_type + '.png')
    plt.savefig(save_path, dpi=600)


if __name__ == "__main__":
    nCustomer, nReplications = 10000, 300
    simulation(nCustomer, nReplications)