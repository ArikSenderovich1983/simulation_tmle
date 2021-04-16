import numpy as np
import pandas as pd
import os

def naive_calculations(df_t1, df_t23, df_t45, cutdown):
    los_list = []
    df_list = [df_t1, df_t23, df_t45]

    for df_t in df_list:
        los = []
        for j in range(len(df_t)):
            if df_t['Consult'][j] == "Yes":
                los.append(df_t['sojourn_time(minutes)'][j] * cutdown)
            else:
                los.append(df_t['sojourn_time(minutes)'][j])
        los_list.append(los)

    mean, median, stdev, P90 = [], [], [], []
    for l in range(len(los_list)):
        mean.append(round(np.mean(los_list[l]), 2))
        median.append(round(np.median(los_list[l]), 2))
        stdev.append(round(np.std(los_list[l]), 2))
        P90.append(round(np.percentile(los_list[l], q=90), 2))

    return mean, median, stdev, P90


if __name__ == "__main__":
    df = pd.read_excel(os.path.join(os.getcwd(), "NYGH_cleaned_test_data.xlsx"), engine='openpyxl')
    nPatientsTotal = 8000
    df = df.iloc[0:nPatientsTotal]
    df_t1 = df[df['Triage Category'] == 'Resuscitation'].reset_index()
    df_t23 = df[df['Triage Category'] == 'Emergent & Urgent'].reset_index()
    df_t45 = df[df['Triage Category'] == 'Less Urgent & Non-Urgent'].reset_index()

    nPatients = [len(df_t1), len(df_t23), len(df_t45)]
    nConsultPatients = [len(df_t1[df_t1['Consult'] == "Yes"]), len(df_t23[df_t23['Consult'] == "Yes"]),
                        len(df_t45[df_t45['Consult'] == "Yes"])]
    percentConsult = [round(nConsultPatients[0] / nPatients[0] * 100, 1),
                      round(nConsultPatients[1] / nPatients[1] * 100, 1),
                      round(nConsultPatients[2] / nPatients[2] * 100, 1)]

    df_results = pd.DataFrame(
        columns=['nPatients [T1, T23, T45]', 'nConsultPatients [T1, T23, T45]', 'percentConsult [T1, T23, T45]',
                 'Cut Down by (%)', 'Mean [T1, T23, T45]', 'Median [T1, T23, T45]',
                 'Stdev [T1, T23, T45]', '90th Percentile [T1, T23, T45]'])

    cutdown_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, cutdown in enumerate(cutdown_percentage):
        print("Consult Patients shorten by {} Percent".format((1-cutdown)*100))
        mean, median, stdev, P90 = naive_calculations(df_t1, df_t23, df_t45, cutdown)
        df_results.loc[i] = nPatients, nConsultPatients, percentConsult, (
                    1 - cutdown) * 100, mean, median, stdev, P90

    save_path_results = os.path.join(os.getcwd(), "Consult_Reduction_Results_Naive.csv")
    df_results.to_csv(save_path_results, index=False, header=True)


