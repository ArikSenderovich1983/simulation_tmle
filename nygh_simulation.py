from nygh_pre_process import *
from datetime import timedelta
import random
import itertools
from sklearn.cluster import KMeans
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import heapq as hq
import time
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


def build_models(df):
    print("\nP(Consult | X) model...")
    df_p_consult = get_df_p_consult(df)
    df_p_consult_train, df_p_consult_test = split_train_test(df_p_consult)
    df_p_consult_train, df_p_consult_test = get_dummy_vars(df_p_consult_train, df_p_consult_test)
    x_features = ['Age Category_Children',
                  'Age Category_Seniors', 'Age Category_Youth', 'Gender Code_M',
                  'Triage Category_Less Urgent & Non-Urgent',
                  'Triage Category_Resuscitation', 'Ambulance_Yes', 'Initial Zone_Disaster', 'Initial Zone_ER Beds',
                  'Initial Zone_GZ',
                  'Initial Zone_HH', 'Initial Zone_OF', 'Initial Zone_Red',
                  'Initial Zone_Resus', 'Initial Zone_SA', 'Initial Zone_U']
    y_features = ['Consult_Yes']
    x_train, x_test, y_train, y_test = get_x_and_y_df(df_p_consult_train, df_p_consult_test, x_features, y_features)

    model_consult_RF = RandomForestClassifier(min_samples_leaf=30, max_depth=None, n_estimators=100, random_state=0)
    model_consult_RF.fit(x_train, y_train)
    threshold = 0.5
    consult_preds = np.where(model_consult_RF.predict_proba(x_train)[:, 1] > threshold, 1, 0).tolist()
    consult_preds.extend(np.where(model_consult_RF.predict_proba(x_test)[:, 1] > threshold, 1, 0))
    consult_preds = pd.Series(consult_preds).replace({1: 'Yes', 0: 'No'}).tolist()

    print("\nP(LOS | X, NIS, consult_pred) model...")
    df_los = df[['Age Category', 'Gender Code', 'Triage Category', 'Ambulance', 'NIS Upon Arrival', 'Consult',
                'Initial Zone', 'patient_arrival_times', 'sojourn_time(minutes)', 'arrival_hour', 'arrival_year',
                'arrival_day_of_week']]
    df_los['consult'] = consult_preds
    df_los_train, df_los_test = split_train_test(df_los)
    df_los_train, df_los_test = get_dummy_vars(df_los_train, df_los_test)
    los_x_features = ['Age Category_Children', 'Age Category_Seniors',
                      'Age Category_Youth', 'Gender Code_M',
                      'Triage Category_Less Urgent & Non-Urgent',
                      'Triage Category_Resuscitation', 'Ambulance_Yes',
                      'Initial Zone_Disaster', 'Initial Zone_ER Beds', 'Initial Zone_GZ',
                      'Initial Zone_HH', 'Initial Zone_OF', 'Initial Zone_Red',
                      'Initial Zone_Resus', 'Initial Zone_SA', 'Initial Zone_U',
                      'Initial Zone_YZ', 'NIS Upon Arrival', 'consult_Yes',
                      'arrival_hour_1', 'arrival_hour_2', 'arrival_hour_3', 'arrival_hour_4',
                      'arrival_hour_5', 'arrival_hour_6', 'arrival_hour_7', 'arrival_hour_8',
                      'arrival_hour_9', 'arrival_hour_10', 'arrival_hour_11',
                      'arrival_hour_12', 'arrival_hour_13', 'arrival_hour_14',
                      'arrival_hour_15', 'arrival_hour_16', 'arrival_hour_17',
                      'arrival_hour_18', 'arrival_hour_19', 'arrival_hour_20',
                      'arrival_hour_21', 'arrival_hour_22', 'arrival_hour_23',
                      'arrival_day_of_week_1', 'arrival_day_of_week_2',
                      'arrival_day_of_week_3', 'arrival_day_of_week_4',
                      'arrival_day_of_week_5', 'arrival_day_of_week_6']
    los_y_features = ['sojourn_time(minutes)']
    los_x_train, los_x_test, los_y_train, los_y_test = get_x_and_y_df(df_los_train, df_los_test, los_x_features,
                                                                      los_y_features)
    model_los_RF = RandomForestRegressor(min_samples_leaf=30, max_depth=None, n_estimators=100, random_state=0)
    model_los_RF.fit(los_x_train, los_y_train)

    return x_train, x_test, y_train, y_test, model_consult_RF, los_x_train, los_x_test, los_y_train, los_y_test, model_los_RF


def sample_from_RF_classifier(model, sample):
    p_consult_prob = model.predict_proba(np.array(sample).reshape(1, -1))  # [P(no_consult), P(yes_consult))
    return np.random.choice([0, 1], 1, p=p_consult_prob[0])


# def sample_from_RF_regressor_helper(model, x_train):
#     n_trees = len(model.estimators_)
#     errors = []
#     # for i in range(n_trees):
#     i=0
#     los_tree_pred = model.estimators_[i].predict(np.array(x_train))
#     leaf_nodes = model.estimators_[i].apply(x_train)
#     for node in leaf_nodes[0:30]:
#         tree_val = model.estimators_[i].tree_.value[node][0,0]
#         errors.extend(los_tree_pred - tree_val)
#     # print(errors)
#     return errors
#
#
# def sample_from_RF_regressor(model, sample, errors):
#     los_RF_pred = model.predict(np.array(sample).reshape(1, -1))
#     error = np.random.choice(errors, 1)
#     if (los_RF_pred[0] + error[0]) < 0:
#         print('los_RF_pred + error < 0')
#         return 0.0
#     return (los_RF_pred[0] + error[0])

def sample_from_RF_regressor(model, sample, errors):
    los_RF_pred = model.predict(np.array(sample).reshape(1, -1))
    error = np.random.choice(errors, 1)
    while (los_RF_pred + error) < 0:
        error = np.random.choice(errors, 1)
    return los_RF_pred + error


def sample_from_RF_regressor_helper(model, x_train, y_train):
    y_preds = model.predict(np.array(x_train))
    y_preds = pd.Series(data=y_preds, index=x_train.index)

    y_train = pd.Series(data=y_train, index=x_train.index)

    condition_triage1 = (x_train['Triage Category_Less Urgent & Non-Urgent'] == 0) & (
                x_train['Triage Category_Resuscitation'] == 1)
    condition_triage23 = (x_train['Triage Category_Less Urgent & Non-Urgent'] == 0) & (
                x_train['Triage Category_Resuscitation'] == 0)
    condition_triage45 = (x_train['Triage Category_Less Urgent & Non-Urgent'] == 1) & (
                x_train['Triage Category_Resuscitation'] == 0)

    return get_errors(x_train, y_train, y_preds, condition_triage1), get_errors(x_train, y_train, y_preds,
                                                                                condition_triage23), get_errors(x_train,
                                                                                                                y_train,
                                                                                                                y_preds,
                                                                                                                condition_triage45)


def get_errors(x_train, y_train, y_preds, condition):
    idx_list = x_train.index[condition]
    actual = y_train[idx_list]
    predicted = y_preds[idx_list]
    errors = []
    for i in range(len(actual)):
        err = actual.iloc[i] - predicted.iloc[i]
        errors.append(err)
    return errors


def simulate(nruns, cutdown, initial_event_calendar, consult_model, consult_x_test, model_los_RF, los_x_test, los_errors_t1, los_errors_t23, los_errors_t45):
    print('sampling LOS errors...')
    np.random.seed(3)  # set random seed
    los_tr_nruns = []
    n_T1, n_T23, n_T45 = 0, 0, 0
    n_consult_T1, n_consult_T23, n_consult_T45 = 0, 0, 0

    for r in range(nruns):
        print('running simulation run #: ' + str(r + 1))
        event_calendar = initial_event_calendar.copy()

        los_tr = [[] for _ in range(3)]  # [[triage1], [triage23], [triage45]]

        curr_nis = 0
        # departure_times = []
        while len(list(event_calendar)) > 0:
            # take an event from the event_calendar
            ts_, event_, id_ = hq.heappop(event_calendar)

            # arrival event happens, need to check if servers are available
            if event_ == 'a':
                curr_nis += 1  # update current number of customers in the system
                if id_ % 1000 == 0:
                    print('arrival: ', id_)

                # sampling consult + LOS
                consult = sample_from_RF_classifier(consult_model, consult_x_test.iloc[id_])
                los_x_test.loc[los_x_test.index[id_], 'consult_Yes'] = consult
                los_x_test.loc[los_x_test.index[id_], 'NIS Upon Arrival'] = curr_nis

                sample = los_x_test.iloc[id_]

                if (sample['Triage Category_Less Urgent & Non-Urgent'] == 0) & (sample['Triage Category_Resuscitation'] == 1):
                    los = sample_from_RF_regressor(model_los_RF, sample, los_errors_t1)
                    if sample['consult_Yes'] == 1:
                        los = los * cutdown
                        if r == 0: n_consult_T1 += 1
                    triage_idx = 0
                    if r == 0: n_T1 += 1

                elif (sample['Triage Category_Less Urgent & Non-Urgent'] == 0) & (sample['Triage Category_Resuscitation'] == 0):
                    los = sample_from_RF_regressor(model_los_RF, sample, los_errors_t23)
                    if sample['consult_Yes'] == 1:
                        los = los * cutdown
                        if r == 0: n_consult_T23 += 1
                    triage_idx = 1
                    if r == 0: n_T23 += 1

                elif (sample['Triage Category_Less Urgent & Non-Urgent'] == 1) & (sample['Triage Category_Resuscitation'] == 0):
                    los = sample_from_RF_regressor(model_los_RF, sample, los_errors_t45)
                    if sample['consult_Yes'] == 1:
                        los = los * cutdown
                        if r == 0: n_consult_T45 += 1
                    triage_idx = 2
                    if r == 0: n_T45 += 1

                los_tr[triage_idx].append(los[0])

                # add a departure event to the event_calendar
                d = ts_ + timedelta(minutes = int(los))
                # departure_times.append(d)
                hq.heappush(event_calendar, (d, 'd', id_))

            # departure event happens
            else:
                curr_nis -= 1  # update current number of customers in the system

        los_tr_nruns.append(los_tr)

    nPatients = [n_T1, n_T23, n_T45]
    nConsultPatients = [n_consult_T1, n_consult_T23, n_consult_T45]
    percentConsult = [round((n_consult_T1/n_T1)*100, 1), round((n_consult_T23/n_T23)*100, 1), round((n_consult_T45/n_T45)*100, 1)]

    return los_tr_nruns, nPatients, nConsultPatients, percentConsult


def total_performance_measures(los_tr_nruns):
    data_mean_list = compute_per_run_performance(los_tr_nruns, "mean")
    data_median_list = compute_per_run_performance(los_tr_nruns, "median")
    data_stdev_list = compute_per_run_performance(los_tr_nruns, "stdev")
    data_90percentile_list = compute_per_run_performance(los_tr_nruns, "P90")

    mean = [round(x, 2) for x in np.mean(data_mean_list, axis=0)]
    median = [round(x, 2) for x in np.mean(data_median_list, axis=0)]
    stdev = [round(x, 2) for x in np.mean(data_stdev_list, axis=0)]
    P90 = [round(x, 2) for x in np.mean(data_90percentile_list, axis=0)]

    return mean, median, stdev, P90


def compute_per_run_performance(los_tr_nruns, performance):
    sample = los_tr_nruns[0]
    n_list = [len(x) for x in sample]
    return_list = []
    for l in los_tr_nruns:
        temp_list = []

        for idx in range(len(n_list)):

            if performance == "mean":
                temp_list.append(np.mean(l[idx]))
            elif performance == "median":
                temp_list.append(np.median(l[idx]))
            elif performance == "stdev":
                variance = (np.std(l[idx]) ** 2) * (n_list[idx] / (n_list[idx] - 1))
                temp_list.append(np.sqrt(variance))
            elif performance == "P90":
                temp_list.append(np.percentile(l[idx], q=90))  # manual: int(np.ceil(len(c)*0.9))-1

        return_list.append(temp_list)
    return return_list


if __name__ == "__main__":
    df = pre_process_data(filename='NYGH_1_8_v1.csv')
    # df = pd.read_csv(os.path.join(os.getcwd(), "cleaned_data.csv"))
    print('simulation prep...')
    _, df_test = split_train_test(df)
    arrivals = df_test['patient_arrival_times'].tolist()
    x_train, x_test, _, y_test, model_consult_RF, los_x_train, los_x_test, los_y_train, los_y_test, model_los_RF = build_models(df)


    print('simulation start...')
    nRuns = 10
    cutdown_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # cutdown_percentage = [0.5, 0.8, 1.0]
    # cutdown_percentage = [0.5]
    df_results = pd.DataFrame(
        columns=['nPatients [T1, T23, T45]', 'nConsultPatients [T1, T23, T45]', 'percentConsult [T1, T23, T45]',
                 'nRuns', 'Cut Down by (%)', 'Mean [T1, T23, T45]', 'Median [T1, T23, T45]',
                 'Stdev [T1, T23, T45]', '90th Percentile [T1, T23, T45]', 'Time to Run (mins)'])
    df_los_data = pd.DataFrame(columns=['Cut Down by (%)', 'LOS Data'])

    los_errors_t1, los_errors_t23, los_errors_t45 = sample_from_RF_regressor_helper(model_los_RF, los_x_train, los_y_train)

    initial_event_calendar = [(a, 'a', i) for i, a in enumerate(arrivals[0:8000])]  # reduced to test out
    hq.heapify(initial_event_calendar)

    for i, cutdown in enumerate(cutdown_percentage):
        print("Consult Patients shorten by {} Percent".format((1-cutdown)*100))
        start = time.time()
        los_tr_nruns, nPatients, nConsultPatients, percentConsult = simulate(nRuns, cutdown, initial_event_calendar,
                                                                             model_consult_RF, x_test, model_los_RF,
                                                                             los_x_test,
                                                                             los_errors_t1, los_errors_t23,
                                                                             los_errors_t45)
        # print(los_tr_nruns)
        mean, median, stdev, P90 = total_performance_measures(los_tr_nruns)
        # print(mean, median, stdev, P90)
        end = time.time()
        time_to_run = round((end - start)/60, 3)
        print(time_to_run)
        df_los_data.loc[i] = (1-cutdown)*100, los_tr_nruns
        df_results.loc[i] = nPatients, nConsultPatients, percentConsult, nRuns, (1-cutdown)*100, mean, median, stdev, P90, time_to_run

    save_path_los_data = os.path.join(os.getcwd(), "Consult_Reduction_LOS_Data.csv")
    df_los_data.to_csv(save_path_los_data, index=False, header=True)
    save_path_results = os.path.join(os.getcwd(), "Consult_Reduction_Results.csv")
    df_results.to_csv (save_path_results, index = False, header=True)

    # x = np.arange(len(departure_times))
    # print('plotting start...')
    # plt.plot(x, departure_times, label='simulated')
    # plt.plot(x, df['Left ED DateTime'].tolist(), label='actual')
    # plt.legend()
    # plt.show()