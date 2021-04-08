import numpy as np
import pandas as pd
import os
from datetime import datetime
import heapq as hq

def pre_process_data(filename):
    print("Pre-processing Data...")
    # Read in data and keep only relevant columns
    filepath = os.path.join(os.getcwd(), filename)
    df = pd.read_csv(filepath)
    df = df[['Age (Registration)','Gender Code','Arrival Mode','Ambulance Arrival DateTime','Triage DateTime',
             'Triage Code','Left ED DateTime','Initial Zone','Consult Service Description (1st)',
             'Diagnosis Code Description','CACS Cell Description']]

    print("Before Pre-processing (relevant columns), df.columns: ", df.columns)
    print("Before Pre-processing (relevant columns), df.shape: ", df.shape)

    df = df[df['Age (Registration)'].notna()]  # drop columns with null values

    # Categorize 'Age'
    df.loc[(df['Age (Registration)'] >= 0) & (df['Age (Registration)'] <= 14), 'Age Category'] = 'Children'
    df.loc[(df['Age (Registration)'] >= 15) & (df['Age (Registration)'] <= 24), 'Age Category'] = 'Youth'
    df.loc[(df['Age (Registration)'] >= 25) & (df['Age (Registration)'] <= 64), 'Age Category'] = 'Adult'
    df.loc[df['Age (Registration)'] >= 65, 'Age Category'] = 'Seniors'
    df.drop(columns=['Age (Registration)'], inplace=True)

    # Drop rows with unknown gender (no missing values for gender after removing missing values for 'Age'
    df = df[df['Gender Code'] != 'U']

    # Triage score (get rid of 9.0; categorize)
    df = df[df['Triage Code'] != 9.0]
    df.loc[df['Triage Code'] == 1, 'Triage Category'] = 'Resuscitation'
    df.loc[(df['Triage Code'] == 2) | (df['Triage Code'] == 3), 'Triage Category'] = 'Emergent & Urgent'
    df.loc[(df['Triage Code'] == 4) | (df['Triage Code'] == 5), 'Triage Category'] = 'Less Urgent & Non-Urgent'
    df.drop(columns=['Triage Code'], inplace=True)

    # Ambulance: "Yes" if "Ambulance Arrival DateTime" column is not null, "No" otherwise
    df.loc[df['Ambulance Arrival DateTime'].isnull() == True, 'Ambulance'] = 'No'
    df.loc[df['Ambulance Arrival DateTime'].isnull() == False, 'Ambulance'] = 'Yes'


    # Consult: "Yes" if "Consult Service Description (1st)" column is not null, "No" otherwise¶
    df.loc[df['Consult Service Description (1st)'].isnull() == True, 'Consult'] = 'No'
    df.loc[df['Consult Service Description (1st)'].isnull() == False, 'Consult'] = 'Yes'

    print("Converting DateTime formats...")
    # Convert column values to DateTime format
    df['Ambulance Arrival DateTime'] = df.apply(
        lambda row: check_float_and_convert_to_dateimte(row['Ambulance Arrival DateTime']), axis=1)
    df['Triage DateTime'] = df.apply(lambda row: check_float_and_convert_to_dateimte(row['Triage DateTime']), axis=1)
    df['Left ED DateTime'] = df.apply(lambda row: check_float_and_convert_to_dateimte(row['Left ED DateTime']), axis=1)

    df['patient_arrival_times'] = df.apply(
        lambda row: max_arrival_time(row['Ambulance Arrival DateTime'], row['Triage DateTime']), axis=1)

    # Add a new column "sojourn_times(minutes)" by calculating (left ED datetime - patient arrival times)
    df['sojourn_time(minutes)'] = df.apply(
        lambda row: (row['Left ED DateTime'] - row['patient_arrival_times']).seconds // 60, axis=1)

    # Fill in missing values with "U" as unknown for initial zone
    df['Initial Zone'].fillna(value='U', inplace=True)

    df = df.reset_index()

    print("Computing NIS...")
    # Create a new column in the dataframe for occupancy/NIS
    '''
        Add arrival & departure events to event calendar, organize events based on timestamp. 
        Keep a counter called curr_nis. 
            If event is arrival, curr_nis+1 and the arriving patient's NIS upon arrival is set to this curr_nis+1. 
            If event is departure, curr_nis-1.
    '''
    df['NIS Upon Arrival'] = 0
    arrival_times = df['patient_arrival_times'].tolist()
    departure_times = df['Left ED DateTime'].tolist()

    arrival_events = [(arrival_time, patient_number + 1, 'a') for patient_number, arrival_time in
                      enumerate(arrival_times)]
    departure_events = [(departure_time, patient_number + 1, 'd') for patient_number, departure_time in
                        enumerate(departure_times)]

    event_calendar = arrival_events + departure_events
    hq.heapify(event_calendar)

    curr_nis = 0
    while (len(event_calendar) != 0):
        timestamp, patient_number, event_type = hq.heappop(event_calendar)
        if event_type == 'a':
            curr_nis += 1
            df.at[patient_number - 1, 'NIS Upon Arrival'] = curr_nis
        elif event_type == 'd':
            curr_nis -= 1

    df['arrival_hour'] = df.apply(lambda row: row['patient_arrival_times'].hour, axis=1)
    df['arrival_day_of_week'] = df.apply(lambda row: row['patient_arrival_times'].weekday(), axis=1)
    df['arrival_year'] = df.apply(lambda row: row['patient_arrival_times'].year, axis=1)
    df.set_index('index', inplace=True)

    df['Age Category'] = df['Age Category'].astype("category")
    df['Gender Code'] = df["Gender Code"].astype("category")
    df['Triage Category'] = df['Triage Category'].astype("category")
    df['Ambulance'] = df["Ambulance"].astype("category")
    df['Consult'] = df["Consult"].astype("category")
    df['arrival_hour'] = df["arrival_hour"].astype("category")
    df['arrival_day_of_week'] = df["arrival_day_of_week"].astype("category")
    df['Initial Zone'] = df["Initial Zone"].astype("category")

    print("After Pre-processing, df.columns: ", df.columns)
    print("After Pre-processing, df.shape: ", df.shape)

    return df


def get_df_p_consult(df_all):
    df_p_consult = df_all[['Age Category', 'Gender Code', 'Triage Category', 'Ambulance', 'Consult', 'Initial Zone',
                           'patient_arrival_times', 'sojourn_time(minutes)', 'NIS Upon Arrival',
                            'arrival_hour', 'arrival_year']]
    return df_p_consult


# Convert relevant columns to DateTime format
def check_float_and_convert_to_dateimte(input_val):
    if type(input_val) != float:
        return datetime.strptime(input_val, "%Y-%m-%d %H:%M:%S")


# Add a new column "patient_arrival_times" by taking MAX(ambulance arrival datetime, triage datetime)
def max_arrival_time(ambulance, triage):
    if type(ambulance) == type(pd.NaT):
        return triage
    else:
        return max(ambulance, triage)


def split_train_test(df_all):
    df_train = df_all[(df_all['arrival_year'] == 2016) | (df_all['arrival_year'] == 2017)]
    df_test = df_all[df_all['arrival_year'] == 2018]
    print('df_train.shape, df_test.shape: ', df_train.shape, df_test.shape)
    return df_train, df_test


def get_dummy_vars(df_train, df_test):
    df_train = pd.get_dummies(data=df_train, drop_first=True)
    df_test = pd.get_dummies(data=df_test, drop_first=True)
    print('df_train_dummies.shape, df_test_dummies.shape: ', df_train.shape, df_test.shape)
    return df_train, df_test


def get_x_and_y_df(df_train, df_test, x_cols, y_cols):
    x_train = df_train[x_cols]
    x_test = df_test[x_cols]
    y_train = df_train[y_cols]
    y_test = df_test[y_cols]
    y_train_reshaped = np.array(y_train).reshape(len(y_train), )  # Consult_yes
    y_test_reshaped = np.array(y_test).reshape(len(y_test), )  # Consult_yes

    print('x_train.shape, x_test.shape: ', x_train.shape, x_test.shape)
    print('y_train.shape, y_test.shape: ', y_train.shape, y_test.shape)
    return x_train, x_test, y_train_reshaped, y_test_reshaped


# if __name__ == "__main__":
#     df = pre_process_data(filename='NYGH_1_8_v1.csv')
#     save_path = os.path.join(os.getcwd(), "cleaned_data.csv")
#     df.to_csv (save_path, index = False, header=True)

