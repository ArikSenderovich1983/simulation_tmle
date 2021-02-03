import pandas as pd
from scipy.stats import bernoulli, expon
import numpy as np
from pycox.datasets import metabric
from pycox.models import LogisticHazard
# from pycox.models import PMF
# from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import improper_gamma
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


from random import choice

from scipy import stats
#from keras import optimizers
#from keras import metrics
#import livelossplot
#plot_losses = livelossplot.PlotLossesKeras()

from matplotlib import pyplot as plt

import glob
from enum import *



def compute_lindley(t_i, new_si):

    #known Lindley

    new_di = []
    new_yi = []

    cur_di = 0
    #new_si_lag = [new_si[i] - new_si[i - 1] for i in range(1, len(new_si))]
    #new_si_lag.insert(0, 0)
    new_si.insert(0,0)
    for j in range(1,len(new_si)-1):
        temp_di = max(0, cur_di + new_si[j-1] - t_i[j])
        new_di.append(temp_di)
        new_yi.append(temp_di + new_si[j-1])  # new_si[j])
        cur_di = temp_di

    mean_yi = np.mean(new_yi)
    #print("LOS after change: " + str(mean_yi))
    var_y = pow(np.sum([pow(y - mean_yi, 2) for y in new_yi]) / (len(new_yi) - 1), 0.5)
    #print("STDEV of LOS after change: " + str(var_y))


    mean_di = np.mean(new_di)
    #print("Waiting after change: " + str(mean_di))
    var_d = pow(np.sum([pow(d - mean_di, 2) for d in new_di]) / (len(new_di) - 1), 0.5)
    #print("STDEV of LOS after change: " + str(var_d))


    return new_yi, new_di, var_y, var_d
def speedup_empirical_service_model(a_i, s_i):
    #we need to fit a KDE model per a_i:
    unique_a = set(a_i)
    dict_s= {}
    #collect all observations of s such that it happens under a
    for a in unique_a:
        dict_s[a] = [s for j,s in enumerate(s_i) if a_i[j]==a ]


    return dict_s
def generate_empirical_services(a_i, dict_s):
    #we need to fit a KDE model per a_i:

    new_si = []
    for n_a in a_i:

        new_si.append(choice(dict_s[n_a]))
    return new_si
def format_func(x, loc):
    if x == 0:
        return '0'
    elif x == 1:
        return 'h'
    elif x == -1:
        return '-h'
    else:
        return '%ih' % x


def speedup_KDE_service_model(a_i, s_i):
    plt_flag = 1
    #we need to fit a KDE model per a_i:
    unique_a = set(a_i)
    log_s_i = np.log(s_i)
    dict_kde_s= {}
    #collect all observations of s such that it happens under a
    for a in unique_a:
        samples = [s for j,s in enumerate(log_s_i) if a_i[j]==a ]

        # Fit KDE (cross-validation used!)
        model_kde = KernelDensity(kernel = 'gaussian', bandwidth=0.1)
        #model_kde = KernelDensity(kernel = 'exponential', bandwidth=0.1)

        #params = {'bandwidth': [0.1+i/10 for i in range(0,9)]}
        #grid = GridSearchCV(KernelDensity(kernel = 'epanechnikov'), params, verbose = 2, n_jobs = 4)
        #grid.fit(np.array(samples).reshape(-1,1))
        #model_kde = grid.best_estimator_
        model_kde.fit(np.array(samples).reshape(-1,1))

        if plt_flag:
            #X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
            x_grid = np.linspace(-4.5, 3.5, 1000)

            pdf = np.exp(model_kde.score_samples(x_grid[:, None]))

            fig, ax = plt.subplots()
            ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % model_kde.bandwidth)
            ax.hist(samples, 30, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
            ax.legend(loc='upper left')
            ax.set_xlim(-4.5, 3.5)


            plt.show()
        dict_kde_s[a] = model_kde


        # Plot original data vs. resampled

        #plt.show()





    return dict_kde_s

def generate_parametric_services(new_ai, sol_m1, sol_m2):
    # ti are the same
    new_si = []
    for a_new in new_ai:
        if a_new == 1:
            new_si.append(expon.rvs(scale=1 / sol_m2))
        else:
            new_si.append(expon.rvs(scale=1 / sol_m1))
    #print("New average S_i: ", np.mean(new_si))

    return new_si
def generate_kde_services(kde_s, ai):
    new_si = []

    for n_a in ai:
        new_si.append(np.exp(kde_s[n_a].sample(1)[0][0]))


    mean_0 = 0
    mean_1 = 0
    count_0 = 0
    count_1 = 0
    check_flag=0
    if check_flag:
        check = [(ai[i], new_si[i]) for i in range(len(ai))]
        for c in check:
            if c[0] == 0:
                mean_0 += c[1]
                count_0 += 1
            else:
                mean_1 += c[1]
                count_1 += 1
        # print(new_si)
        print(float(1 / np.mean(new_si)))
        print(mean_0 / count_0)
        print(mean_1 / count_1)

    return new_si
def generate_arrivals(sol_l, n_customers):
    # ti are the same
    new_ti = [expon.rvs(scale=1 / sol_l) for i in range(n_customers)]

    return new_ti
def generate_intervention(p, n_customers):
    # ti are the same
    new_ai = [bernoulli.rvs(p) for i in range(n_customers)]

    return new_ai

def MLE_known(a_i,t_i,s_i):
    sol_p = np.sum(a_i)/len(a_i)
    sol_l = len(t_i)/np.sum(t_i)

    sol_m1 = np.sum([1-a for a in a_i])/np.sum([s for j,s in enumerate(s_i) if a_i[j]==0])
    sol_m2 = np.sum(a_i)/np.sum([s for j,s in enumerate(s_i) if a_i[j]==1])

    print(sol_p, sol_l, sol_m1, sol_m2)
    return sol_p, sol_l, sol_m1, sol_m2

def return_params(df):
    df['T'] = 0.0

    for i in range(len(df)):
        if i == 0:
            df.at[i, 'T'] = df.at[i, 'arrival_time']
        else:
            df.at[i, 'T'] = df.at[i, 'arrival_time'] - df.at[i - 1, 'arrival_time']


    a_i = np.array(df['A'])
    s_i = np.array(df['S'])
    t_i = np.array(df['T'])
    w_i = np.array(df['elapsed']) - np.array(df['S'])

    return a_i, t_i, s_i, w_i

def update_Data(data, **kwargs):
    data['p_0'].append(kwargs.get('p_speed'))

    data['mu_2'].append(kwargs.get('mu_2'))
    data['p_1'].append(kwargs.get('p_1'))

    data['run'].append(kwargs.get('run'))
    data['lambda'].append(kwargs.get('lam_'))

    data['method'].append(kwargs.get('method'))


    data['cf_los'].append(kwargs.get('avg_los'))
    data['cf_wait'].append(kwargs.get('avg_wait'))

    data['cf_std_los'].append(kwargs.get('std_los'))
    data['cf_std_wait'].append(kwargs.get('std_wait'))



    data['real_los'].append(kwargs.get('real_los'))
    data['real_std_los'].append(kwargs.get('real_std_los'))

    data['real_wait'].append(kwargs.get('real_wait'))
    data['real_std_wait'].append(kwargs.get('real_std_wait'))



def create_lindley_df(t_i, s_i, w_i, a_i):
    a_lag = [0]
    a_lag.extend([a_i[i - 1] for i in range(1, len(a_i))])
    w_prev = [0]
    w_prev.extend([w_i[i - 1] for i in range(1, len(w_i))])
    s_prev = [0]
    s_prev.extend([s_i[i - 1] for i in range(1, len(s_i))])
    # w_lindley = [max(0,w_prev[i]+(s_prev[i]-t_i[i])) for i in range(0,len(w_i))]
    dict_w = {"T_i": t_i, "S_lag": s_prev, "W_lag": w_prev, "W_target": w_i}  # , "W_lindley": w_lindley}
    df_train = pd.DataFrame.from_dict(dict_w)
    return df_train

def create_service_df(a_i, s_i):

    # w_lindley = [max(0,w_prev[i]+(s_prev[i]-t_i[i])) for i in range(0,len(w_i))]
    dict_s = {"S_i": s_i, "A_i": a_i}  # , "W_lindley": w_lindley}
    df = pd.DataFrame.from_dict(dict_s)
    return df






from datetime import *
import keras_lib as kl

data = {}
data['p_0'] = []

data['mu_2'] = []
data['p_1'] = []
data['run'] = []
data['lambda'] = []

data['method'] = []
#data['cf_los'] = []
data['cf_wait'] = []
#data['cf_std_los'] = []
data['cf_std_wait'] = []


#data['real_los'] = []
#data['real_std_los'] = []

data['real_wait'] = []
data['real_std_wait'] = []

#for file in glob.glob("datafiles/*.csv"):#["datafiles\\out_[LAMBDA,MU,P_OBSERVABILITY,MAX_ARRIVALS](0.84,1.0,0.10,120000)[preprocessed].csv"]:#
#for file in ["datafiles\\interventfion_data_0.9_2_0.9.csv"]:
mle_experiment = {'lambda': [], 'mu_2': [], 'p_0': [], 'p_1':[],
                  #'avg_cf_los': [],
                  'avg_cf_wait': [],
                  #'stdev_cf_los': [],
                  'stdev_cf_wait': [],
                  #'avg_sim_los': [],
                  'avg_sim_wait': [], #'stdev_sim_los': [],
                  'stdev_sim_wait': [],
                  'step': []}
for file in glob.glob("datafiles\\experimental_data\\*.csv"):
    #csvfiles.append(file)
    print("Working on file ",str(file) )
    print('Start time: ', datetime.now())
    filename = file.split("datafiles\\")[1]
    split_file = filename.split("intervention_data_")[1].split(".csv")[0].split("_")
    setting = '_'.join(split_file)
    lam_ = float(split_file[0])

    mu_2 = float(split_file[1])
    p_speed = float(split_file[2])

    if p_speed ==0 or p_speed==1:
        continue

    #for method in method_list:


        #filename = "intervention_mu_2.csv"
    K=1 #increase to 30
    n_rows = 990000
    n_min = 10000
    #run_list = list(range(0,K))
    #p_list = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1] #[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    p_list = [0]
    #p_list = [0.7]
    avg_yi = {}
    avg_di = {}
    # todo: change to tuples and kwargs for main function
    df = pd.read_csv(file, nrows=n_rows)

    method = 'UnknownL'
    approach='SemiParam'


    # 'OoBox' # 'ML'

    # option 1: parametric, known L

    # if False:
    # sol_p, sol_l, sol_m1, sol_m2 = MLE_(a_i, t_i, s_i)
    a_i, t_i, s_i, w_i = return_params(df)

    #mg1_speedup_KDE_service_model(a_i,a_i,s_i)
    sol_p, sol_l, sol_m1, sol_m2 = MLE_known(a_i, t_i, s_i)
    if approach=='Param':
        print('Parametric approach: ', sol_p, sol_l, sol_m1, sol_m2)
    else:
        #kde_s = speedup_KDE_service_model(a_i, s_i)
        kde_s = speedup_empirical_service_model(a_i,s_i)
    #approximating Lindley is our current only solution
    if method!='KnownL':
        df_lindley = create_lindley_df(t_i, s_i, w_i, a_i)
        regressor_lindley, sc_X = kl.approx_Lindley(df_lindley,target_col="W_target", num_cols=3)


    #Construct confidence intervals for intervention:

    steps = 10
    for p_1 in p_list:
        #if p_1 == p_speed:
            #continue

        for s in range(steps):
            print('current p_1 and step:', p_1, ',', s)
            g_comp_wi = -3
            if method=='KnownL':


                boot_ai = generate_intervention(sol_p, n_rows)
                boot_ti = generate_arrivals(sol_l, n_rows)

                if approach == 'Param':
                    boot_si = generate_parametric_services(boot_ai, sol_m1, sol_m2)
                    sol_p_boot, sol_l_boot, sol_m1_boot, sol_m2_boot = MLE_known(boot_ai, boot_ti, boot_si)

                else:
                    boot_si = generate_empirical_services(boot_ai, kde_s)#generate_kde_services(kde_s, boot_ai)
                    sol_p_boot, sol_l_boot, sol_m1_boot, sol_m2_boot = MLE_known(boot_ai, boot_ti, boot_si)
                    kde_s_boot = speedup_empirical_service_model(boot_ai, boot_si)
                    #kde_s_boot = speedup_KDE_service_model(boot_ai, boot_si)

                #g_comp:
                print('G-COMP!')
                g_comp_ai = generate_intervention(p_1, n_rows)
                g_comp_ti = generate_arrivals(sol_l_boot, n_rows)
                if approach == 'Param':
                    g_comp_si = generate_parametric_services(g_comp_ai, sol_m1_boot, sol_m2_boot)
                else:
                    #g_comp_si = generate_kde_services(kde_s_boot, g_comp_ai)
                    g_comp_si = generate_empirical_services(g_comp_ai, kde_s_boot)

                _, g_comp_wi, _, std_w = compute_lindley(g_comp_ti, g_comp_si)


                print('Estimated Si: '+str(np.mean(s_i)))
                print('Bootstrap Si: '+str(np.mean(boot_si)))
                print('GComp Si: '+str(np.mean(g_comp_si)))




            elif method=='UnknownL':


                boot_ai = generate_intervention(sol_p, n_rows)
                boot_ti = generate_arrivals(sol_l, n_rows)

                if approach == 'Param':
                    boot_si = generate_parametric_services(boot_ai, sol_m1, sol_m2)
                    sol_p_boot, sol_l_boot, sol_m1_boot, sol_m2_boot = MLE_known(boot_ai, boot_ti, boot_si)

                else:
                    boot_si = generate_empirical_services(boot_ai, kde_s)
                    sol_p_boot, sol_l_boot, sol_m1_boot, sol_m2_boot = MLE_known(boot_ai, boot_ti, boot_si)
                    #kde_s_boot = speedup_KDE_service_model(boot_ai, boot_si)
                    kde_s_boot = speedup_empirical_service_model(boot_ai, boot_si)
                boot_wi, std_w = kl.pred_keras_lindley(boot_si, boot_ti, regressor_lindley, sc_X)

                df_lindley_boot = create_lindley_df(boot_ti, boot_si, boot_wi[1:], boot_ai)
                boot_regressor_lindley, sc_X_boot = kl.approx_Lindley(df_lindley_boot, target_col="W_target", num_cols=3)



                # g_comp:
                print('G-COMP!')
                g_comp_ai = generate_intervention(p_1, n_rows)
                g_comp_ti = generate_arrivals(sol_l_boot, n_rows)
                if approach == 'Param':
                    g_comp_si = generate_parametric_services(g_comp_ai, sol_m1_boot, sol_m2_boot)
                else:
                    #g_comp_si = generate_kde_services(kde_s_boot, g_comp_ai)
                    g_comp_si = generate_empirical_services(g_comp_ai, kde_s_boot)

                print('Estimated Si: ' + str(np.mean(s_i)))
                print('Bootstrap Si: ' + str(np.mean(boot_si)))
                print('GComp Si: ' + str(np.mean(g_comp_si)))

                try:
                    g_comp_wi, std_w = kl.pred_keras_lindley(g_comp_si, g_comp_ti, boot_regressor_lindley, sc_X_boot)
                except ValueError:
                    g_comp_wi = [-1]*(n_min+1)
                    std_w = -1
                    print('System exploded')
                except TypeError:
                    g_comp_wi = [-2]*(n_min+1)
                    std_w = -2
                    print('Convergence Issue')

            mle_experiment['lambda'].append(lam_)
            mle_experiment['mu_2'].append(mu_2)
            mle_experiment['p_0'].append(p_speed)
            mle_experiment['p_1'].append(p_1)
            # mle_experiment['avg_cf_los'].append(np.mean(known_yi[n_min:]))
            mle_experiment['avg_cf_wait'].append(np.mean(g_comp_wi[n_min:]))
            # mle_experiment['stdev_cf_los'].append(std_y)
            mle_experiment['stdev_cf_wait'].append(std_w)
            df_cf = pd.read_csv(
                "datafiles\\on_hold\\intervention_data_" + str(lam_) + "_" + str(int(mu_2)) + "_" + str(p_1) + ".csv")

            #real_los = np.mean(df_cf.loc[n_min:, 'elapsed'].values)
            real_wait = np.mean(df_cf.loc[n_min:, 'elapsed'] - df_cf.loc[n_min:, 'S'])
            #real_std_los = np.std(df_cf.loc[n_min:, 'elapsed'].values)
            real_std_wait = np.std(df_cf.loc[n_min:, 'elapsed'] - df_cf.loc[n_min:, 'S'])


            #mle_experiment['avg_sim_los'].append(real_los)
            mle_experiment['avg_sim_wait'].append(real_wait)
            #mle_experiment['stdev_sim_los'].append(real_std_los)
            mle_experiment['stdev_sim_wait'].append(real_std_wait)
            mle_experiment['step'].append(s)
            print('Data vs. predicted: ', real_wait, np.mean(g_comp_wi[n_min:]))



    df_dict = pd.DataFrame.from_dict(mle_experiment)
    df_dict.to_csv('counterfactual_results_'+str(approach)+'_'+str(method)+ '_' + str(n_rows) + '_customers.csv')

