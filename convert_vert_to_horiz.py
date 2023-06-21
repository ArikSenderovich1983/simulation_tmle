#assuming a, s, d order per customer (no abandonments)
import pandas as pd
import matplotlib.pyplot as plt

df_vert = pd.read_csv('simulation_result_run_0.csv')
df_vert.sort_values(by = ['customer', 'timestamp','event'], inplace=True)   #assume d>s
df_vert.reset_index(inplace=True, drop=True)

#customer	event	timestamp	server	class	nis	niq

#df_horiz['case_id'] = df_vert['customer'].copy(deep=True)
horiz_dict = {'case':[], 'class':[], 'arrival':[], 'service':[], 'departure':[], 'server':[], 'arr_nis':[], 'arr_niq':[], 'sojourn':[], 'cum_arrival':[], 'arr_nservice':[]}
all_classes = df_vert['class'].unique()
context_names = ['x1', 'x2']
for c in all_classes:
    horiz_dict['arr_nis_class_'+str(c)] = []
for c in context_names:
    horiz_dict[c] = []
cur_id = df_vert.at[0,'customer']
cur_server = df_vert.at[0,'server']
cur_class = df_vert.at[0,'class']
arr_nis = df_vert.at[0,'nis']
arr_niq = df_vert.at[0,'niq']
cum_arrival = 1
timestamps = [df_vert.at[0, 'timestamp']]
nservice = df_vert.at[0,'nservice']
nis_vec = {}
for c in all_classes:
    nis_vec[str(c)]  = df_vert.at[0,'nis_class_'+str(c)]

for i in range(1,len(df_vert)):
    if cur_id==df_vert.at[i,'customer']:
        timestamps.append(df_vert.at[i,'timestamp'])
        cur_server = df_vert.at[i, 'server']
    else:
        horiz_dict['case'].append(cur_id)
        horiz_dict['arrival'].append(timestamps[0])
        horiz_dict['service'].append(timestamps[1])
        horiz_dict['departure'].append(timestamps[2])
        horiz_dict['server'].append(cur_server)
        horiz_dict['class'].append(cur_class)
        horiz_dict['arr_nis'].append(arr_nis)
        horiz_dict['arr_niq'].append(arr_niq)
        horiz_dict['sojourn'].append(timestamps[2]-timestamps[0])
        horiz_dict['cum_arrival'].append(cum_arrival)
        horiz_dict['arr_nservice'].append(nservice)
        for c in context_names:
            horiz_dict[c].append(df_vert.at[i, c])
        for c in df_vert['class'].unique():
            horiz_dict['arr_nis_class_' + str(c)].append(nis_vec[str(c)])


        cur_id = df_vert.at[i, 'customer']
        cur_server = df_vert.at[i, 'server']
        cur_class = df_vert.at[i, 'class']
        timestamps = [df_vert.at[i, 'timestamp']]
        arr_nis = df_vert.at[i, 'nis']
        arr_niq = df_vert.at[i, 'niq']
        nis_vec = {}
        for c in all_classes:
            nis_vec[str(c)] = df_vert.at[i, 'nis_class_' + str(c)]
        nservice = df_vert.at[i, 'nservice']
        cum_arrival+=1



df_horiz =pd.DataFrame.from_dict(horiz_dict)

df_horiz['last_remaining_los'] = df_horiz['sojourn'].shift(1) - (df_horiz['arrival'] - df_horiz['arrival'].shift(1))
df_horiz.at[df_horiz.index[0], 'last_remaining_los'] = 0

df_horiz.to_csv('horizontal_multi_many.csv', index=False)

