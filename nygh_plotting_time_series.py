import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def plot(df, plot_name):
    arrivals, departures, avg_nis = [], [], []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)


    if plot_name == 'hour':
        n = 24
        fig.set_size_inches(24, 12)
    elif plot_name == 'day_of_week':
        n = 7
        fig.set_size_inches(12, 12)

    x = np.linspace(0, n, n+1)

    for i in range(n):
        arrivals.append(round(len(df[df['arrival_{}'.format(plot_name)] == i]) / 365))
        departures.append(round(len(df[df['departure_{}'.format(plot_name)] == i]) / 365))
        df_i = df[df['arrival_{}'.format(plot_name)] == i]
        avg_nis.append(round(np.sum(df_i['NIS Upon Arrival']) / len(df_i)))

    ax1.plot(x[1:], arrivals, color='b', label='arrivals_by_{}'.format(plot_name), linestyle='-', marker='o')
    ax2.plot(x[1:], departures, color='g', label='departures_by_{}'.format(plot_name), linestyle='-', marker='o')
    ax3.plot(x[1:], avg_nis, color='r', label='avg_nis_by_{}'.format(plot_name), linestyle='-', marker='o')
    ax1.set(ylabel='Arrivals by {}'.format(plot_name), xlabel='{}'.format(plot_name))
    ax2.set(ylabel='Departures by {}'.format(plot_name), xlabel='{}'.format(plot_name))
    ax3.set(ylabel='Average NIS Upon Arrival by {}'.format(plot_name), xlabel='{}'.format(plot_name))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    if plot_name == 'hour':
        ax1.set_xtick(x)
        ax2.set_xtick(x)
        ax3.set_xtick(x)
    elif plot_name == 'day_of_week':
        days = ['','Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        ax1.set_xticklabels(days)
        ax2.set_xticklabels(days)
        ax3.set_xticklabels(days)

    fig.suptitle('Time Series ({})'.format(plot_name))

    save_path = os.path.join(os.getcwd(), 'NYGH_' + plot_name + '.png')
    plt.savefig(save_path, dpi=600)



if __name__ == "__main__":
    data = pd.read_excel(os.path.join(os.getcwd(), "NYGH_cleaned_test_data.xlsx"), engine='openpyxl')
    data['departure_hour'] = data.apply(lambda row: row['Left ED DateTime'].hour, axis=1)
    data['departure_day_of_week'] = data.apply(lambda row: row['Left ED DateTime'].weekday(), axis=1)
    plot(data, plot_name='hour')
    plot(data, plot_name='day_of_week')