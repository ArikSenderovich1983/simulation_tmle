from q_intervention import *
from scipy.stats import t, sem
import matplotlib.pyplot as plt

def simulation(nCustomer, nReplications):
    """
    Four types of performance measures: Wait Time in Queue, Time in System, Number in Queue, Number in System
    """
    # Input parameters
    lam = 1  # fixed arrival rate
    mu = 1.1  # fixed original service rate
    mu_prime_list = np.array([2, 2.5, 3])
    p_intervention_list = np.linspace(0, 1, 11)
    confidence_level = 0.95

    # Calculate the value of rho, run the simulation only if rho < 1
    # rho = lam / mu
    # if rho >= 1:
    #     return

    # Keep track of statistics, 3 values of mu, 11 values of P(intervention)
    # All classes
    tis_mean, wiq_mean, niq_mean, nis_mean = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_ci, wiq_ci, niq_ci, nis_ci = np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f'), np.zeros((3,11),dtype='f,f')
    tis_90per, wiq_90per, niq_90per, nis_90per = np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11)), np.zeros((3,11))
    tis_90per_ci, wiq_90per_ci, niq_90per_ci, nis_90per_ci = np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f'), np.zeros((3, 11), dtype='f,f')

    # Individual Classes
    nClasses = 2
    nRow = len(mu_prime_list) * nClasses
    tis_mean_by_class, wiq_mean_by_class, niq_mean_by_class, nis_mean_by_class = np.zeros(
        (nRow, 11)), np.zeros((nRow, 11)), np.zeros(
        (nRow, 11)), np.zeros((nRow, 11))
    tis_mean_ci_by_class, wiq_mean_ci_by_class, niq_mean_ci_by_class, nis_mean_ci_by_class = np.zeros(
        (nRow, 11), dtype='f,f'), np.zeros((nRow, 11), dtype='f,f'), np.zeros(
        (nRow, 11), dtype='f,f'), np.zeros((nRow, 11), dtype='f,f')
    tis_90per_by_class, wiq_90per_by_class, niq_90per_by_class, nis_90per_by_class = np.zeros(
        (nRow, 11)), np.zeros((nRow, 11)), np.zeros(
        (nRow, 11)), np.zeros((nRow, 11))
    tis_90per_ci_by_class, wiq_90per_ci_by_class, niq_90per_ci_by_class, nis_90per_ci_by_class = np.zeros(
        (nRow, 11), dtype='f,f'), np.zeros(
        (nRow, 11), dtype='f,f'), np.zeros((nRow, 11), dtype='f,f'), np.zeros((nRow, 11), dtype='f,f')

    for i, mu_prime in enumerate(mu_prime_list):
        for j, p_interv in enumerate(p_intervention_list):
            print("Parameters: mu_prime={}, p_intervention={}".format(mu_prime, p_interv))
            # q_ = multi_class_single_station_fcfs(lambda_ = lam, classes = [0], probs = [1.0],
            #              mus = [mu], prob_speedup=[p_interv], mus_speedup=[mu_prime],
            #              servers = 1)
            q_ = multi_class_single_station_fcfs(lambda_=lam, classes=[0, 1], probs=[0.5, 0.5],
                                                 mus=[mu, mu], prob_speedup=[p_interv, 0.0], mus_speedup=[mu_prime, mu_prime],
                                                 servers=1)

            q_.simulate_q(customers=nCustomer, runs=nReplications)

            # Time in System
            los_mean_all, los_mean_all_ci, los_mean_percentile90_all, los_mean_percentile90_all_ci, los_means, los_means_ci, los_mean_percentile90s, los_mean_percentile90s_ci = los_wiq(
                q_, confidence_level, nReplications, los=True)
            tis_mean[i, j], tis_ci[i, j] = los_mean_all, los_mean_all_ci
            tis_90per[i, j], tis_90per_ci[i, j] = los_mean_percentile90_all, los_mean_percentile90_all_ci
            offset = 0
            for k in range(len(los_means)):
                tis_mean_by_class[i + offset, j]  = los_means[k]
                tis_mean_ci_by_class[i + offset, j] = (los_means_ci[0][k], los_means_ci[1][k])
                tis_90per_by_class[i + offset, j]  = los_mean_percentile90s[k]
                tis_90per_ci_by_class[i + offset, j] = (los_mean_percentile90s_ci[0][k], los_mean_percentile90s_ci[1][k])
                offset += len(mu_prime_list)

            # Wait Time in Queue
            wiq_mean_all, wiq_mean_all_ci, wiq_mean_percentile90_all, wiq_mean_percentile90_all_ci, wiq_means, wiq_means_ci, wiq_mean_percentile90s, wiq_mean_percentile90s_ci = los_wiq(
                q_, confidence_level, nReplications, los=False)
            wiq_mean[i, j], wiq_ci[i, j] = wiq_mean_all, wiq_mean_all_ci
            wiq_90per[i, j], wiq_90per_ci[i, j] = wiq_mean_percentile90_all, wiq_mean_percentile90_all_ci
            offset = 0
            for k in range(len(los_means)):
                wiq_mean_by_class[i + offset, j] = wiq_means[k]
                wiq_mean_ci_by_class[i + offset, j] = (wiq_means_ci[0][k], wiq_means_ci[1][k])
                wiq_90per_by_class[i + offset, j]  = wiq_mean_percentile90s[k]
                wiq_90per_ci_by_class[i + offset, j] = (wiq_mean_percentile90s_ci[0][k], wiq_mean_percentile90s_ci[1][k])
                offset += len(mu_prime_list)

            # Number in Queue
            niq_num_all_classes, niq_num_all_classes_ci, niq_mean_percentile90_all, niq_mean_percentile90_all_ci, niq_nums, niq_nums_ci, niq_mean_percentile90s, niq_mean_percentile90s_ci = niq_nis(
                q_, confidence_level, nReplications, queue=True)
            niq_mean[i, j], niq_ci[i, j] = niq_num_all_classes, niq_num_all_classes_ci
            niq_90per[i, j], niq_90per_ci[i, j] = niq_mean_percentile90_all, niq_mean_percentile90_all_ci
            offset = 0
            for k in range(len(los_means)):
                niq_mean_by_class[i + offset, j] = niq_nums[k]
                niq_mean_ci_by_class[i + offset, j] = (niq_nums_ci[0][k], niq_nums_ci[1][k])
                niq_90per_by_class[i + offset, j] = niq_mean_percentile90s[k]
                niq_90per_ci_by_class[i + offset, j] = (niq_mean_percentile90s_ci[0][k], niq_mean_percentile90s_ci[1][k])
                offset += len(mu_prime_list)

            # Number in System
            nis_num_all_classes, nis_num_all_classes_ci, nis_mean_percentile90_all, nis_mean_percentile90_all_ci, nis_nums, nis_nums_ci, nis_mean_percentile90s, nis_mean_percentile90s_ci = niq_nis(
                q_, confidence_level, nReplications, queue=False)
            nis_mean[i, j], nis_ci[i, j] = nis_num_all_classes, nis_num_all_classes_ci
            nis_90per[i, j], nis_90per_ci[i, j] = nis_mean_percentile90_all, nis_mean_percentile90_all_ci
            offset = 0
            for k in range(len(los_means)):
                nis_mean_by_class[i + offset, j] = nis_nums[k]
                nis_mean_ci_by_class[i + offset, j] = (nis_nums_ci[0][k], nis_nums_ci[1][k])
                nis_90per_by_class[i + offset, j] = nis_mean_percentile90s[k]
                nis_90per_ci_by_class[i + offset, j] = (nis_mean_percentile90s_ci[0][k], nis_mean_percentile90s_ci[1][k])
                offset += len(mu_prime_list)

    # Plotting Graphs (All Classes)
    plot_mean_90percentile_with_CI(tis_mean, tis_ci, tis_90per, tis_90per_ci,
                                   "Time in System With 95% Confidence Interval, {} Customers, {} Replications, {} Classes (All)".format(
                                       nCustomer, nReplications, nClasses))
    plot_mean_90percentile_with_CI(wiq_mean, wiq_ci, wiq_90per, wiq_90per_ci,
                                   "Wait Time in Queue With 95% Confidence Interval, {} Customers, {} Replications, {} Classes (All)".format(
                                       nCustomer, nReplications, nClasses))
    plot_mean_90percentile_with_CI(niq_mean, niq_ci, niq_90per, niq_90per_ci,
                                   "Number in Queue With 95% Confidence Interval, {} Customers, {} Replications, {} Classes (All)".format(
                                       nCustomer, nReplications, nClasses))
    plot_mean_90percentile_with_CI(nis_mean, nis_ci, nis_90per, nis_90per_ci,
                                   "Number in System With 95% Confidence Interval, {} Customers, {} Replications, {} Classes (All)".format(
                                       nCustomer, nReplications, nClasses))

    # Plotting Graphs (Individual Classes)
    start, end = 0, len(mu_prime_list)
    for g in range(nClasses):
        plot_mean_90percentile_with_CI(tis_mean_by_class[start:end], tis_mean_ci_by_class[start:end],
                                       tis_90per_by_class[start:end], tis_90per_ci_by_class[start:end],
                                       "Time in System With 95% Confidence Interval, {} Customers, {} Replications, Class #{}".format(
                                           nCustomer, nReplications, g+1))
        plot_mean_90percentile_with_CI(wiq_mean_by_class[start:end], wiq_mean_ci_by_class[start:end],
                                       wiq_90per_by_class[start:end], wiq_90per_ci_by_class[start:end],
                                       "Wait Time in Queue With 95% Confidence Interval, {} Customers, {} Replications, Class #{}".format(
                                           nCustomer, nReplications, g+1))
        plot_mean_90percentile_with_CI(niq_mean_by_class[start:end], niq_mean_ci_by_class[start:end],
                                       niq_90per_by_class[start:end], niq_90per_ci_by_class[start:end],
                                       "Number in Queue With 95% Confidence Interval, {} Customers, {} Replications, Class #{}".format(
                                           nCustomer, nReplications, g+1))
        plot_mean_90percentile_with_CI(nis_mean_by_class[start:end], nis_mean_ci_by_class[start:end],
                                       nis_90per_by_class[start:end], nis_90per_ci_by_class[start:end],
                                       "Number in System With 95% Confidence Interval, {} Customers, {} Replications, Class #{}".format(
                                           nCustomer, nReplications, g+1))
        start += len(mu_prime_list)
        end += len(mu_prime_list)


def los_wiq(queueing_system, confidence_level, n, los=False):

    if los:
        tracker = queueing_system.get_los_tracker()
    else:
        tracker = queueing_system.get_wait_time_tracker()

    avg_time_by_class, avg_time_all_classes = [], []
    time_percentile90_by_class, time_percentile90_all_classes = [], []
    for rep in tracker:
        # Expected Values
        avg_time_classes, time_percentile90_classes = [], []
        time_list_all_classes, time_percentile90_list_all_classes = [], []
        for time_list in rep:
            avg_time_classes.append(np.mean(time_list))
            time_list_all_classes = np.concatenate([time_list_all_classes, time_list])

        avg_time_by_class.append(avg_time_classes)  # Individual Classes
        avg_time_all_classes.append(np.mean(time_list_all_classes))  # All Classes

        # 90th Percentile
        for time_percentile90_list in rep:
            time_percentile90_classes.append(np.percentile(time_percentile90_list, 90))
            time_percentile90_list_all_classes = np.concatenate([time_percentile90_list_all_classes, time_percentile90_list])
        time_percentile90_by_class.append(time_percentile90_classes)  # Individual Classes
        time_percentile90_all_classes.append(np.percentile(time_percentile90_list_all_classes, 90))   # All Classes

    # Individual Classes - 1) expected value with CI, 2) 90th percentile with CI
    time_mean_by_class, time_mean_by_class_ci = compute_mean_and_CI(avg_time_by_class, confidence_level, n,
                                                                  all_classes=False)
    time_mean_percentile90_by_class, time_mean_percentile90_by_class_ci = compute_mean_and_CI(time_percentile90_by_class,
                                                                                            confidence_level, n,
                                                                                            all_classes=False)
    # time_mean_by_class_var = ((np.std(avg_time_by_class, axis=0))**2)*(n/(n - 1))
    # time_mean_by_class_sd = np.sqrt(time_mean_by_class_var)

    # All Classes - 1) expected value with CI, 2) 90th percentile with CI
    time_mean_all, time_mean_all_ci = compute_mean_and_CI(avg_time_all_classes, confidence_level, n,
                                                                        all_classes=True)
    time_mean_percentile90_all, time_mean_percentile90_all_ci = compute_mean_and_CI(time_percentile90_all_classes,
                                                                                    confidence_level, n,
                                                                                    all_classes=True)

    return time_mean_all, time_mean_all_ci, time_mean_percentile90_all, time_mean_percentile90_all_ci, \
           time_mean_by_class, time_mean_by_class_ci, time_mean_percentile90_by_class, time_mean_percentile90_by_class_ci


def niq_nis(queueing_system, confidence_level, n, queue=False):
    classes = queueing_system.get_classes()
    if queue:
        tracker = queueing_system.get_queue_tracker()
    else:
        tracker = queueing_system.get_nis_tracker()
    avg_num_list_by_class, avg_num_list_all_classes = [], []
    percentile90_list_by_class, percentile90_list_all_classes = [], []
    for rep in tracker:
        t_n = rep[0][-1][0]
        area_all_classes = 0
        num_list_all_classes = []  # [n1, n2, n3, ...] N's are not unique
        num_sets_by_class = []  # [[n1, n2, n3, ...], [n1, n2, n3, ...], ...]
        num_pmf_list_by_class = []  # normalized probabilities [[p1, p2, p3, ...], [p1, p2, p3, ...], ...]

        avg_num_list_classes = []  # [rep1_c1_mean, rep1_c2_mean, ...]
        percentile90_by_class = []  # [rep1_c1_90thperc, rep1_c2_90thperc, ...]

        for c in classes:
            num_list_all_classes += [tup[1] for tup in rep[c]]  # list of N's for customers of all classes
        num_set_all_classes = list(set(num_list_all_classes))  # unique list of N's for all classes
        num_pmf_all_classes = np.zeros(len(num_set_all_classes))

        for c in classes:
            class_x_tracker = rep[c]

            area_by_class = 0

            num_list_class_x = [tup[1] for tup in class_x_tracker]  # list of N's for customers of class x
            num_set_class_x = list(set(num_list_class_x))  # unique list of N's for customers of class x
            num_sets_by_class.append(num_set_class_x)  # append to sets
            num_pmf_by_class = np.zeros(len(num_set_class_x))  # initialize pmf with probabilities of 0

            for i in range(1, len(class_x_tracker)):
                t_i1, t_i0 = class_x_tracker[i][0], class_x_tracker[i-1][0]
                Num_i1 = class_x_tracker[i][1]

                # E(NIQ) = sum_i=1_to_n_ [(t_i - t_i-1)*NIQ_i] / t_n; E(NIS) = sum_i=1_to_n_ [(t_i - t_i-1)*NIS_i] / t_n
                area_by_class += (t_i1 - t_i0) * Num_i1

                # For 90th percentile by class
                idx = num_sets_by_class[c].index(Num_i1)  # find the index for N = Num_i1
                num_pmf_by_class[idx] += t_i1 - t_i0  # accumulate time units for N = n1, n2, ... for each class
                # For 90th percentile of all classes
                idx = num_set_all_classes.index(Num_i1)
                num_pmf_all_classes[idx] += t_i1 - t_i0

            # For expected values
            avg_num_list_classes.append(area_by_class / t_n)
            area_all_classes += area_by_class

            # For 90th percentile by class
            num_pmf_class_x = num_pmf_by_class / t_n  # normalize pmf for customers in class x
            num_pmf_list_by_class.append(num_pmf_class_x)

            cum_prob, n, done = 0, 0, False
            while not done:
                cum_prob += num_pmf_class_x[n]
                if cum_prob >= 0.9:
                    percentile90_by_class.append(num_set_class_x[n])
                    done = True
                n += 1

        avg_num_list_by_class.append(avg_num_list_classes)  # [[rep1_c1_mean, rep1_c2_mean, ...], [rep2_c1_mean, rep2_c2_mean, ...]]
        avg_num_list_all_classes.append(area_all_classes / t_n)  # [rep1_all_mean, rep2_all_mean, ...]

        percentile90_list_by_class.append(percentile90_by_class)  # [[rep1_c1_90thperc, rep1_c2_90thperc, ...], [rep2_c1_90thperc, rep2_c2_90thperc, ...], ...]

        num_pmf_all_classes = num_pmf_all_classes / (t_n * len(classes))  # normalize pmf for customers in all classes
        cum_prob_all, n_all, done_all = 0, 0, False
        while not done_all:
            cum_prob_all += num_pmf_all_classes[n_all]
            if cum_prob_all >= 0.9:
                percentile90_list_all_classes.append(num_set_all_classes[n_all])  # [rep1_all_90thperc, rep2_all_90thperc, ...]
                done_all = True
            n_all += 1

    # Individual Classes - 1) expected value with CI, 2) 90th percentile with CI
    num_mean_by_class, mean_num_by_class_ci = compute_mean_and_CI(avg_num_list_by_class, confidence_level, n,
                                                                        all_classes=False)
    num_mean_percentile90_by_class, num_mean_percentile90_by_class_ci = compute_mean_and_CI(percentile90_list_by_class,
                                                                                            confidence_level, n,
                                                                                            all_classes=False)

    # All Classes - 1) expected value with CI, 2) 90th percentile with CI
    num_mean_all_classes, num_mean_all_classes_ci = compute_mean_and_CI(avg_num_list_all_classes, confidence_level, n,
                                                                        all_classes=True)
    num_mean_percentile90_all_classes, num_mean_percentile90_all_classes_ci = compute_mean_and_CI(
        percentile90_list_all_classes, confidence_level, n, all_classes=True)

    return num_mean_all_classes, num_mean_all_classes_ci, num_mean_percentile90_all_classes, \
           num_mean_percentile90_all_classes_ci, num_mean_by_class, mean_num_by_class_ci, \
           num_mean_percentile90_by_class, num_mean_percentile90_by_class_ci


def compute_mean_and_CI(data, confidence_level, n, all_classes=False):
    if all_classes:
        data_mean = np.mean(data)
        data_mean_stderr = sem(data)
        if data_mean_stderr == 0:  # all data points are equal
            data_mean_ci = (data_mean, data_mean)
        else:
            data_mean_ci = t.interval(confidence_level, n - 1, data_mean, data_mean_stderr)
    else:
        data_mean = np.mean(data, axis=0)
        data_mean_stderr = sem(data, axis=0)
        data_mean_ci = t.interval(confidence_level, n - 1, data_mean, data_mean_stderr)
    return data_mean, data_mean_ci


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

    ax1.plot(x, y1, 'b', label='mu_prime=2')
    ax1.fill_between(x, ci1_lower, ci1_upper, color='b', alpha=0.1)
    ax1.plot(x, y2, 'g', label='mu_prime=2.5')
    ax1.fill_between(x, ci2_lower, ci2_upper, color='g', alpha=0.1)
    ax1.plot(x, y3, 'r', label='mu_prime=3')
    ax1.fill_between(x, ci3_lower, ci3_upper, color='r', alpha=0.1)
    ax1.set(xlabel = 'P(speedup)', ylabel = 'Expected Value')
    # ax1.legend(('mu_prime=2', 'mu_prime=2.5', 'mu_prime=3'))
    ax1.legend()

    # Plotting 90th Percentiles
    y90p1, y90p2, y90p3 = percentile90[0], percentile90[1], percentile90[2]
    ci90p1_lower, ci90p1_upper = [percentile90_ci[0][i][0] for i in range(len(percentile90_ci[0]))], [
        percentile90_ci[0][i][1] for i in range(len(percentile90_ci[0]))]
    ci90p2_lower, ci90p2_upper = [percentile90_ci[1][i][0] for i in range(len(percentile90_ci[1]))], [
        percentile90_ci[1][i][1] for i in range(len(percentile90_ci[0]))]
    ci90p3_lower, ci90p3_upper = [percentile90_ci[2][i][0] for i in range(len(percentile90_ci[2]))], [
        percentile90_ci[2][i][1] for i in range(len(percentile90_ci[0]))]

    ax2.plot(x, y90p1, 'b', label='mu_prime=2')
    ax2.fill_between(x, ci90p1_lower, ci90p1_upper, color='b', alpha=0.1)
    ax2.plot(x, y90p2, 'g', label='mu_prime=2.5')
    ax2.fill_between(x, ci90p2_lower, ci90p2_upper, color='g', alpha=0.1)
    ax2.plot(x, y90p3, 'r', label='mu_prime=3')
    ax2.fill_between(x, ci90p3_lower, ci90p3_upper, color='r', alpha=0.1)
    ax2.set(xlabel='P(speedup)', ylabel='90th Percentile')
    # ax2.legend(('mu_prime=2', 'mu_prime=2.5', 'mu_prime=3'))
    ax2.legend()

    save_path = os.path.join(os.getcwd(), plot_type + '.png')
    plt.savefig(save_path, dpi=600)


if __name__ == "__main__":
    nCustomer, nReplications = 10000, 30
    simulation(nCustomer, nReplications)