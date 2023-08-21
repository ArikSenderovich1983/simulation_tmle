import pandas as pd
import os
from datetime import datetime
import heapq as hq
import itertools
import holidays

def read_file(filename):
    filepath = os.path.join(os.getcwd(), filename)
    df = pd.read_csv(filepath)
    return df

def check_float_and_convert_to_datetime(input_val):
    """
    Converts a given string to DateTime format
    """
    if type(input_val) != float:
        return datetime.strptime(input_val, "%m/%d/%Y %H:%M")

def max_arrival_time(ambulance, triage):
    """
    Adds a new column "patient_arrival_times" by taking MAX(ambulance arrival datetime, triage datetime)
    """
    if type(ambulance) == type(pd.NaT):
        return triage
    else:
        return max(ambulance, triage)

def clean_data(df):

    # drop rows with null values
    df = df[df['Age (Registration)'].notna()]  
    df = df[df['Gender Code'].notna()]

    # fill unknowns
    df['Initial Zone'].fillna(value='U', inplace=True)
    
    # Categorize 'Age'
    df.loc[(df['Age (Registration)'] >= 0) & (df['Age (Registration)'] <= 14), 'Age Category'] = 'Children'
    df.loc[(df['Age (Registration)'] >= 15) & (df['Age (Registration)'] <= 24), 'Age Category'] = 'Youth'
    df.loc[(df['Age (Registration)'] >= 25) & (df['Age (Registration)'] <= 64), 'Age Category'] = 'Adult'
    df.loc[df['Age (Registration)'] >= 65, 'Age Category'] = 'Seniors'
    df.drop(columns=['Age (Registration)'], inplace=True)

    # drop unknown gender columns
    df = df[df['Gender Code'] != 'U']

    # Create ambulance categories
    df.loc[df['Ambulance Arrival DateTime'].isnull() == True, 'Ambulance'] = 'No'
    df.loc[df['Ambulance Arrival DateTime'].isnull() == False, 'Ambulance'] = 'Yes'
    
    # Create consult categories
    df.loc[df['Consult Service Description (1st)'].isnull() == True, 'Consult'] = 'No'
    df.loc[df['Consult Service Description (1st)'].isnull() == False, 'Consult'] = 'Yes'
    df.drop(columns=['Consult Service Description (1st)'], inplace=True)

    # Convert column values to DateTime format
    df['Ambulance Arrival DateTime'] = df.apply(lambda row: check_float_and_convert_to_datetime(row['Ambulance Arrival DateTime']), axis=1)
    df['patient_arrival'] = df.apply(lambda row: check_float_and_convert_to_datetime(row['Triage DateTime']), axis=1)
    df['departure_datetime'] = df.apply(lambda row: check_float_and_convert_to_datetime(row['Left ED DateTime']), axis=1)
    # Define patient arrival times:
    df['arrival_datetime'] = df.apply( lambda row: max_arrival_time(row['Ambulance Arrival DateTime'], row['patient_arrival']), axis=1)
    # Add a new column "sojourn_times(minutes)" by calculating (left ED datetime - patient arrival times)
    df['sojourn'] = df.apply(
        lambda row: (row['departure_datetime'] - row['arrival_datetime']).total_seconds() // 60, axis=1)
    # Select "sojourn_times(minutes)" that are larger than 0
    df = df[df['sojourn'] > 0].reset_index()
    df['cum_arrival'] = df.apply(lambda row: row['index'] + 1, axis=1)
    df.rename(columns={"index": "case"}, inplace = True)
    first_arrival = df['arrival_datetime'].iloc[0]
    df['arrival'] = df.apply( lambda row: (row['arrival_datetime'] - first_arrival).total_seconds() // 60, axis = 1)
    df['departure'] = df.apply( lambda row: (row['departure_datetime'] - first_arrival).total_seconds() // 60, axis = 1)
    
    return df

def create_classes(df):
    temp_list = []
    for _, val in df['Discharge Disposition Description'].items():
        if val.split(" ", 1)[0] == "Admit":
            val = "Admitted"
        else:
            val = "Not Admitted"
        temp_list.append(val)
    df['Admission'] = temp_list
    df.drop(columns=['Discharge Disposition Description'], inplace=True)

    # Categorize patients based on triage code and admission
    # Triage score (get rid of 9.0; categorize)
    df = df[df['Triage Code'] != 9.0]
    # T123: 'Resuscitation, Emergent & Urgent', T45: 'Less Urgent & Non-Urgent'
    df.loc[(df['Admission'] == 'Admitted') & ((df['Triage Code'] == 1) | (df['Triage Code'] == 2) | (
            df['Triage Code'] == 3)), 'class'] = "0"
    df.loc[(df['Admission'] == 'Not Admitted') & ((df['Triage Code'] == 1) | (df['Triage Code'] == 2) | (
            df['Triage Code'] == 3)), 'class'] = "1"
    df.loc[(df['Triage Code'] == 4) | (df['Triage Code'] == 5), 'class'] = "2"

    df = df.astype({"class": int})
    
    return df

def calculate_los_remaining(df):
    df['arrival_shifted'] = df.arrival.shift(-1)
    df['last_remaining_los'] = df.apply( lambda row: max(0, (row['sojourn'] - row['arrival_shifted'] + row['arrival'])), axis = 1)
    df['last_remaining_los'] = df.last_remaining_los.shift(1)
    
    df['last_rem_class_0'] = ''
    df['last_rem_class_1'] = ''
    df['last_rem_class_2'] = ''

    for ind in df.index:
        if ind == 0:
            df['last_rem_class_0'][ind] = 0
            df['last_rem_class_1'][ind] = 0
            df['last_rem_class_2'][ind] = 0
        elif df['class'][ind-1] == 0:
            df['last_rem_class_0'][ind] = max(0, df['departure'][ind-1]-df['arrival'][ind])
            df['last_rem_class_1'][ind] = max(0, df['last_rem_class_1'][ind-1] - df['arrival'][ind] + df['arrival'][ind-1])
            df['last_rem_class_2'][ind] = max(0, df['last_rem_class_2'][ind-1] - df['arrival'][ind] + df['arrival'][ind-1])
        elif df['class'][ind-1] == 1:
            # print(df['departure'][ind-1], df['arrival'][ind])
            df['last_rem_class_1'][ind] = max(0, df['departure'][ind-1]-df['arrival'][ind])
            df['last_rem_class_0'][ind] = max(0, df['last_rem_class_0'][ind-1] - df['arrival'][ind] + df['arrival'][ind-1])
            df['last_rem_class_2'][ind] = max(0, df['last_rem_class_2'][ind-1] - df['arrival'][ind] + df['arrival'][ind-1])
        elif df['class'][ind-1] == 2:
            df['last_rem_class_2'][ind] = max(0, df['departure'][ind-1]-df['arrival'][ind])
            df['last_rem_class_1'][ind] = max(0, df['last_rem_class_1'][ind-1] - df['arrival'][ind] + df['arrival'][ind-1])
            df['last_rem_class_0'][ind] = max(0, df['last_rem_class_0'][ind-1] - df['arrival'][ind] + df['arrival'][ind-1])
            
    df.loc[0, 'last_remaining_los'] = 0
    df.pop('patient_arrival')
    df.pop('arrival_shifted')
    return df

def compute_nis_features(df):
    """
    Computes NIS features for all system states with the following high-level procedure:
        - Add arrival & departure events to event calendar, organize events based on timestamp.
        - Keep a counter called curr_nis.
            - If event is arrival, curr_nis+1 and the arriving patient's NIS upon arrival is set to this curr_nis+1.
            - If event is departure, curr_nis-1.

    @ params:
        df (pd.DataFrame): the DataFrame to add new columns with NIS information for all system states

    @ return:
        df (pd.DataFrame): the DataFrame with NIS features added
    """

    print("Computing NIS...")

    unique_classes = df['class'].unique()

    # Initialize system state 0: general NIS
    df['arr_nis'] = 0
    df['prev_nis_upon_arrival'] = 0
    
    # Initialize system state 1: NIS by patient type
    nis_by_patient_type =()
    prev_nis_by_patient_type = ()
    for _, pt in enumerate(unique_classes):
        df['arr_nis_class_{}'.format(int(pt))] = 0
        nis_by_patient_type += (0,)
        df['prev_nis_upon_arrival_class_{}'.format(int(pt))] = 0
        prev_nis_by_patient_type += (0,)

    arrival_times = df['arrival_datetime'].tolist()
    departure_times = df['departure_datetime'].tolist()

    arrival_times = [(arrival_time, patient_number, 'a') for patient_number, arrival_time in
                     enumerate(arrival_times)]
    departure_times = [(departure_time, patient_number, 'd') for patient_number, departure_time in
                       enumerate(departure_times)]

    event_calendar = arrival_times + departure_times
    hq.heapify(event_calendar)

    curr_nis = 0
    curr_nis_by_pt = nis_by_patient_type
    prev_nis = 0
    prev_nis_by_pt = prev_nis_by_patient_type

    while len(event_calendar) != 0:
        timestamp, id_, event_type = hq.heappop(event_calendar)
        if event_type == 'a':
            # System State = 0
            df.at[id_, 'prev_nis_upon_arrival'] = prev_nis # assign prev customer's NIS upon arrival to current customer
            curr_nis += 1
            df.at[id_, 'arr_nis'] = curr_nis
            prev_nis = curr_nis # update  the "temporary NIS counters for prev customer"

            # System State = 1
            for _, pt in enumerate(unique_classes):
                df.at[id_, 'prev_nis_upon_arrival_class_{}'.format(int(pt))] = prev_nis_by_pt[int(pt)]

            val = df['class'][id_]
            lst = list(curr_nis_by_pt)
            lst[int(val)] = curr_nis_by_pt[int(val)] + 1
            curr_nis_by_pt = tuple(lst)
            
            for _, pt in enumerate(unique_classes):
                df.at[id_, 'arr_nis_class_{}'.format(int(pt))] = curr_nis_by_pt[int(pt)]
            prev_nis_by_pt = curr_nis_by_pt


        elif event_type == 'd':
            # System State = 0
            curr_nis -= 1

            # System State = 1
            val = df['class'][id_]
            lst = list(curr_nis_by_pt)
            lst[int(val)] = curr_nis_by_pt[int(val)] - 1
            curr_nis_by_pt = tuple(lst)


    return df

def rearrange_columns(df):
    # df.drop(['Left ED DateTime', 'Triage DateTime', 'Ambulance Arrival DateTime', 'prev_nis_upon_arrival_class_1', 'prev_nis_upon_arrival_class_0', 'prev_nis_upon_arrival_class_2', 'arrival_datetime', 'departure_datetime'], axis=1)
    df = df[['case', 'class', 'arrival', 'departure', 'arr_nis', 'sojourn', 'cum_arrival', 'arr_nis_class_0', 'arr_nis_class_1', 'arr_nis_class_2', 'last_remaining_los', 'last_rem_class_0', 'last_rem_class_1', 'last_rem_class_2', 'Triage Code', 'Age Category', 'Initial Zone', 'Gender Code', 'Ambulance', 'Consult', 'Admission']]
    
    return df

def save_data(df, filename):
    print('Saving Results...')
    save_path = os.path.join(os.getcwd(), filename[:-4] + "_cleaned.xlsx")
    df.to_excel(save_path, engine='xlsxwriter', index=False)
    df.to_pickle(filename[:-4]+ "_cleaned.pkl")

def change_column_datatypes(df):
    df = df.astype({"Triage Code":'category', "class":'category', "last_rem_class_0": int, "last_rem_class_1": int, "last_rem_class_2": int})
    return df

def main_process(filename):
    # filename = "Trial20.csv"
    df = read_file(filename)
    #remove unecessary columns
    columns = ["Age (Registration)", "Gender Code", "Initial Zone", "Ambulance Arrival DateTime", 
               "Consult Service Description (1st)", "Discharge Disposition Description", "Triage Code",
                 "Triage DateTime", "Left ED DateTime"]
    df = df[columns]

    df = clean_data(df)
    df = create_classes(df)
    df = calculate_los_remaining(df)
    df = compute_nis_features(df)
    df = rearrange_columns(df)
    df = change_column_datatypes(df)
    save_data(df, filename)
    return df




