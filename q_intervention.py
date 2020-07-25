# test push to master

#I am testing whether Nancy can see my changes.

import pandas as pd
from resource import *
import heapq as hq

import numpy as np

import plot_workload as pw
class multi_class_single_station_fcfs:
    #defining the intialization function with the proper parameters.
    def __init__(self, **kwargs):

        self.lambda_ = kwargs.get('lambda_', 1)
        self.classes_ = kwargs.get('classes', [0])
        #probability of arrival being class 1
        #todo: update to uniform distribution (default value)
        #todo: start a GIT for us to work on this.

        self.probs = kwargs.get('probs', [1])
        #pre-intervention mus
        self.mus = kwargs.get('mus',[1])
        self.probs_speedup = kwargs.get('prob_speedup', [0]*len(self.classes_))

        self.mus_speedup = kwargs.get('mus_speedup', self.mus)
        self.servers = kwargs.get('servers', 1)



        self.data = []
        #going to track queues - all start empty (assumption)
        #timestamp, #inQueue
        #keeping track of the statistics:
        self.queue_tracker = [[(0,0)] for c in self.classes_]
        self.los_tracker = [[] for c in self.classes_]
        self.trackers = []
        self.friends = []
        self.sla_levels = []


    def simulate_q(self, customers, runs):


        np.random.seed(3)

        for r in range(runs):
            print('running simulation run #: ' + str(r + 1))

            t_ = 0


            sim_arrival_times = []
            service_times = []
            classes_ = []
            interv_ = []
            for c in range(customers):
                #simulating next arrival time
                sim_arrival_times.append(t_+ Distribution(dist_type=DistributionType.exponential, rate=self.lambda_).sample())

                t_ = sim_arrival_times[len(sim_arrival_times)-1]
                #sampling the class of arriving patient
                c_ = np.random.choice(self.classes_, p=self.probs)
                classes_.append(c_)
                #sampling whether intervention or not

                interv_.append(np.random.choice(np.array([0,1]),
                                                p = np.array([1-self.probs_speedup[c_], self.probs_speedup[c_]])))
                if interv_[len(interv_)-1]==0:
                    service_times.append(Distribution(dist_type=DistributionType.exponential, rate=self.mus[c_]).sample())
                else:
                    service_times.append(
                        Distribution(dist_type=DistributionType.exponential, rate=self.mus_speedup[c_]).sample())



            event_log = []
            self.queue_tracker = [[(0, 0)] for c in self.classes_]
            self.los_tracker = [[] for c in self.classes_]
            #four types of events: arrival, departure = 'a', 'd' queue and service (queue start and service start)
            #every tuple is (timestamp, event_type, customer id, server_id)
            event_calendar = [(a, 'a', i, -1) for i,a in enumerate(sim_arrival_times)]
            hq.heapify(event_calendar)

            queue = []
            hq.heapify(queue)
            #heap is ordered by timestamp - every element is (timestamp, station)
            #need to manage server assignment
            in_service = [0 for s in range(self.servers)]
            server_assignment = [0 for s in range(self.servers)]
            temp_friends = {}
            while len(list(event_calendar))>0:
                #current event - added to the simulation

                ts_, event_, id_, server_ = hq.heappop(event_calendar)
                if event_=='a':
                    #arrival happens, we need to check if servers are available
                    event_log.append((ts_, 'a', id_, interv_[id_], classes_[id_]))
                    if sum(in_service)<self.servers:
                        #there is a room in service:
                        for j,s in enumerate(in_service):
                            if s==0:
                                #assigning server
                                in_service[j] = 1
                                server_assignment[j] = id_
                                hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,j))
                                break

                        event_log.append((ts_, 's', id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))
                        self.los_tracker[classes_[id_]].append(ts_ + service_times[id_] - sim_arrival_times[id_])
                        temp_friends[id_] = []

                    else:
                        #join the queue, no room on servers
                        temp_friends[id_] = [str(q[1])+"_"+str(r) for q in list(queue)]
                        hq.heappush(queue,(ts_, id_))
                        event_log.append((ts_, 'q', id_, interv_[id_], classes_[id_]))
                        self.queue_tracker[classes_[id_]].append((ts_,self.queue_tracker[classes_[id_]][-1][1]+1))
                        for class_ in self.classes_:
                            if classes_[id_]!=class_:
                                self.queue_tracker[class_].append(
                                    (ts_, self.queue_tracker[class_][-1][1]))


                else: #event is departure
                    in_service[server_] = 0
                    #server_assignment[server_] = 0
                    if len(list(queue))>0:
                        _ , id_ = hq.heappop(queue)
                        event_log.append((ts_,'s',id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))
                        #server became available we mount it
                        in_service[server_] = 1
                        server_assignment[server_] = id_
                        hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,server_))
                        self.los_tracker[classes_[id_]].append(ts_ + service_times[id_] - sim_arrival_times[id_])
                        self.queue_tracker[classes_[id_]].append((ts_,self.queue_tracker[classes_[id_]][-1][1]-1))
                        for class_ in self.classes_:
                            if classes_[id_]!=class_:
                                self.queue_tracker[class_].append(
                                    (ts_, self.queue_tracker[class_][-1][1]))


            self.data.append(event_log)
            self.trackers.append((self.los_tracker,self.queue_tracker))
            self.friends.append(temp_friends)

        print('Done simulating...')


    def generate_data(self, **kwargs):
        #generating data for intervention experiments
        sla_q = kwargs.get('sla_', 0.9)
        quant_flag = kwargs.get('quant_flag',True)
        write_file = kwargs.get('write_file', True)
        filename = kwargs.get('filename','intervention')
        offset = 0.0
        self.avg_sla_value=0
        avg_sla_q = 0
        for j,e_l in enumerate(self.data):
            print("Run #"+str(j+1))

            #creating a data-frame to manage the event logs
            #one per simulation run - we will later want to compare interventions
            df = pd.DataFrame(e_l, columns=['timestamp', 'event_type', 'id', 'A', 'C'])
            #two things: we both want plots to see if the simulator makes sense, and create synthetic data
            #print(df.head(5))
            #order by id and timestamp there may be tie between a and q - we don't care
            df.sort_values(by=['id','timestamp'], inplace=True)
            df.reset_index(drop=True,inplace=True)

            df['elapsed'] = 0.0
            df['S'] = 0.0
            df['arrival_time'] = 0.0
            df['id_run'] = ""
            cur_id = df.at[0,'id']
            cur_start = df.at[0,'timestamp']
            df['FriendsID'] = " "
            df['nFriends'] = 0
            temp_friends = self.friends[j]
            for i in range(len(df)):
                df.at[i,'id_run'] = str(df.at[i,'id'])+"_"+str(j)
                if cur_id==df.at[i,'id']:
                    df.at[i, 'arrival_time'] = cur_start + offset
                    df.at[i,'elapsed'] = df.at[i,'timestamp'] - cur_start
                    # departure event:
                    if df.at[i,'event_type']=='d':
                        df.at[i, 'S'] = df.at[i,'timestamp']-df.at[i-1,'timestamp']
                    #print(df.at[i,'event_type'])
                    #input("Press Enter to continue...")


                else:


                    cur_id = df.at[i, 'id']
                    cur_start= df.at[i,'timestamp'].copy()
                    df.at[i,'arrival_time'] = cur_start+offset

                df.at[i,'FriendsID'] = " ".join(map(str, temp_friends[df.at[i,'id']]))
                df.at[i,'nFriends'] = len(temp_friends[df.at[i, 'id']])
            offset = offset+max(df['timestamp'])
            print('Average LOS per run: ')
            print(np.mean(df[df.event_type == 'd']['elapsed']))

            df['SLA'] = 0
            if quant_flag==True:
                sla_ = np.quantile(df[df.event_type == 'd']['elapsed'], q= sla_q)  # np.mean(df['elapsed'])
            else:
                sla_ = sla_q[j]
            self.sla_levels.append(sla_)
            for i in range(len(df)):
                if df.at[i, 'elapsed'] > sla_:
                    df.at[i, 'SLA'] = 1
            print("Sla level:")
            print(sum(df[df.event_type == 'd']['SLA']) / len(df[df.event_type == 'd']))
            if write_file:
                if j==0:
                    #df[df.event_type=='d'].loc[:,['id_run', 'arrival_time', 'event_type','C', 'A', 'elapsed']].to_csv('intervention_data.csv', index=False, header=True)
                    df[df.event_type == 'd'].loc[:,['id_run', 'arrival_time', 'timestamp', 'event_type','C', 'A', 'S', 'FriendsID','nFriends','elapsed', 'SLA']].to_csv(str(filename)+'.csv', index=False, header=True)

                else:
                    #df[df.event_type=='d'].loc[:,['id_run', 'arrival_time', 'event_type','C', 'A', 'elapsed']].to_csv('intervention_data.csv', mode='a', index= False, header=False)
                    df[df.event_type == 'd'].loc[:,['id_run', 'arrival_time', 'timestamp','event_type','C', 'A', 'S','FriendsID','nFriends', 'elapsed','SLA']].to_csv(str(filename)+'.csv', mode='a', index= False, header=False)

        print("Average SLA value: "+str(np.mean(self.sla_levels)))
        return np.mean(self.sla_levels)



    def performance_los(self):


        run_avg_los = 0
        run_avg_los_class = [0 for c in self.classes_]
        for run, tr in enumerate(self.trackers):
            print("Run #"+str(run+1))
            total_q_list = [sum([tr[1][c][i][1] for c in range(len(self.classes_))]) for i in range(len(tr[1][0]))]


            print('Max QL is: '+str(max([sum([tr[1][c][i][1] for c in range(len(self.classes_))]) for i in range(len(tr[1][0]))]  ) ))

            if run==0:
                pw.plot_K_graphs([total_q_list],['total_queue_length'], 'total_q over time','queue_len', ['b'],False,0,[],[])

            total_los_run=[]
            for j,c in enumerate(self.classes_):

                if len(tr[0][c])>0:
                    print("LOS for class " + str(c) + ":")
                    los_c =np.mean([tr[0][c][i]  for i in range(len(tr[0][c])) ])
                    print(los_c)
                    total_los_run.extend(tr[0][c])
                    run_avg_los_class[c]+=los_c
            if len(total_los_run)>0:
                print("LOS for all classes: ")
                avg_run_los = np.mean(total_los_run)
                print(avg_run_los)
                run_avg_los+=avg_run_los
        print("LOS across runs: "+str(run_avg_los/len(self.trackers)))
        for c in self.classes_:
            print("LOS per class "+str(c)+": " + str(run_avg_los_class[c]/len(self.trackers)))

points_ = 20
runs_ = 1
mus_speedup_list = [11] #np.linspace(1.1,11,points_)
sla_list = [3 for i in range(points_)]
customers_ = 10000
sla_results =[]
#todo: write different intervention files per value. Run TMLE and plot. Fix sla at 4.
for j,mu_2 in enumerate(mus_speedup_list):
    print('mu 2 is: ' +str(mu_2))
    q_ = multi_class_single_station_fcfs(lambda_ = 1, classes = [0], probs = [1.0],
                                         mus = [1.1], prob_speedup=[0.3], mus_speedup=[mu_2],
                                         servers = 1)

    q_2 = multi_class_single_station_fcfs(lambda_ = 1, classes = [0], probs = [1.0],
                                         mus = [1.1], prob_speedup=[0.7], mus_speedup=[mu_2],
                                         servers = 1)


    q_.simulate_q(customers = customers_, runs = runs_)

    sla_results.append(q_.generate_data(sla_ = 0.9, quant_flag=True, write_file = False, filename = 'intervention_'+str(j)))

    q_.performance_los()

    #fc.calculate_friends("intervention_data.csv", window_ = 5)


    q_2.simulate_q(customers = customers_, runs = runs_)
    #q_2.generate_data(sla_ = sla_list, quant_flag=False, write_file = False)

    q_2.generate_data(sla_ = q_.sla_levels, quant_flag=False, write_file = False)
    q_2.performance_los()

print(sla_results)