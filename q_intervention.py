import pandas as pd
from resource import *
import heapq as hq
import numpy as np
import plot_workload as pw
import os

class multi_class_single_station_fcfs:
    # defining the queueing system using given parameters
    def __init__(self, **kwargs):

        # initialize parameters
        self.lambda_ = kwargs.get('lambda_', 1)  # arrival rate
        self.classes_ = kwargs.get('classes', [0])  # class types
        #probability of arrival being class 1
        #todo: update to uniform distribution (default value)

        self.probs = kwargs.get('probs', [1])  # probability of arrival for each class
        #pre-intervention mus
        self.mus = kwargs.get('mus',[1])  # service rate without intervention
        self.probs_speedup = kwargs.get('prob_speedup', [0]*len(self.classes_))  # probability of speedup

        self.mus_speedup = kwargs.get('mus_speedup', self.mus)  # service rate with intervention
        self.servers = kwargs.get('servers', 1)  # number of servers

        self.laplace_params = kwargs.get('laplace_params', [0, 0.5])  # location and scale parameters

        # initialize trackers with relevant statistics, assume all start empty
        self.data = []  # event logs
        self.wait_time_tracker = []  # [[[timestamp, timestamp, ...], ...]]
        self.los_tracker = []  # [[[timestamp, timestamp, ...], ...]]
        self.queue_tracker = []  # [[[(timestamp, Nq), (timestamp, Nq), ...], ...]]
        self.nis_tracker = []  # [[[(timestamp, NIS), (timestamp, NIS), ...], ...]]
        # self.trackers = []  # [(los_tracker, queue_tracker), (los_tracker, queue_tracker), ...]
        # self.friends = []
        self.sla_levels = []

    # Getters
    def get_classes(self):
        return self.classes_

    def get_wait_time_tracker(self):
        return self.wait_time_tracker

    def get_los_tracker(self):
        return self.los_tracker

    def get_queue_tracker(self):
        return self.queue_tracker

    def get_nis_tracker(self):
        return self.nis_tracker


    def simulate_q(self, customers, runs, system_type=1):

        np.random.seed(3)  # set random seed

        for r in range(runs):
            print('running simulation run #: ' + str(r + 1))

            t_ = 0  # time starts from zero
            sim_arrival_times = []
            service_times = []
            classes_ = []
            interv_ = []

            for c in range(customers):
                print('Customer: '+str(c))
                # simulate next arrival
                # next arrival time: t_ + inter-arrival_time
                sim_arrival_times.append(
                    t_ + Distribution(dist_type=DistributionType.exponential, rate=self.lambda_).sample())

                # move forward the timestamp
                t_ = sim_arrival_times[len(sim_arrival_times)-1]

                # sampling the class of arriving patient
                c_ = np.random.choice(self.classes_, p=self.probs)
                classes_.append(c_)

                # sampling whether intervention or not
                interv_.append(np.random.choice(np.array([0,1]),
                                                p = np.array([1-self.probs_speedup[c_], self.probs_speedup[c_]])))
                if interv_[len(interv_)-1]==0:
                    service_times.append(Distribution(dist_type=DistributionType.exponential, rate=self.mus[c_]).sample())
                else:
                    service_times.append(
                        Distribution(dist_type=DistributionType.exponential, rate=self.mus_speedup[c_]).sample())

            # system_type (default 1) --> system_type 1 = M/G/1; system_type 2 = G/G/1 (appointment)
            if system_type == 2:
                # next arrival time: a' = a + noise, noise here is punctuality
                # draw a punctuality time from laplace distribution
                for i, arrival_time in enumerate(sim_arrival_times):
                    done = False
                    while not done:
                        punctuality = Distribution(dist_type=DistributionType.laplace, location=self.laplace_params[0],
                                                   scale=self.laplace_params[1]).sample()
                        actual_arrival_time = arrival_time + punctuality
                        if actual_arrival_time >= 0:
                            sim_arrival_times[i] = actual_arrival_time
                            done = True

            event_log = []
            queue_tr = [[(0, 0)] for c in self.classes_]  # [[(timestamp, Nq), (timestamp, Nq), ...], ...]
            nis_tr = [[(0, 0)] for c in self.classes_]  # [[(timestamp, NIS), (timestamp, NIS), ...], ...]
            los_tr = [[] for c in self.classes_]  # [[timestamp, timestamp, ...], ...]
            wait_tr = [[] for c in self.classes_]  # [[timestamp, timestamp, ...], ...]
            # four types of events: arrival, departure = 'a', 'd' queue and service (queue start and service start)
            # every tuple is (timestamp, event_type, customer id, server_id)
            event_calendar = [(a, 'a', i, -1) for i,a in enumerate(sim_arrival_times)]
            hq.heapify(event_calendar)

            queue = []  # [(timestamp, customer_id), (timestamp, customer_id), ...]
            hq.heapify(queue)
            # heap is ordered by timestamp - every element is (timestamp, station)
            # need to manage server assignment
            in_service = [0 for s in range(self.servers)]  # 0 = not in service; 1 = in service
            server_assignment = [0 for s in range(self.servers)]
            # temp_friends = {}

            # keep going if there are still events waiting to occur
            while len(list(event_calendar))>0:
                # take an event from the event_calendar
                print(list(event_calendar)[0])
                ts_, event_, id_, server_ = hq.heappop(event_calendar)

                # arrival event happens, need to check if servers are available
                if event_ == 'a':
                    # log arrival event
                    event_log.append((ts_, 'a', id_, interv_[id_], classes_[id_]))

                    # update nis_tracker - add 1 to the class in which the customer belongs to
                    nis_tr[classes_[id_]].append((ts_, nis_tr[classes_[id_]][-1][1] + 1))

                    # update nis_tracker for all other classes, NIS stays the same
                    for class_ in self.classes_:
                        if classes_[id_] != class_:
                            nis_tr[class_].append((ts_, nis_tr[class_][-1][1]))

                    # if there is a room in service
                    if sum(in_service) < self.servers:
                        for j,s in enumerate(in_service):
                            # find the first available server
                            if s == 0:
                                # set the jth server to be busy, serving customer with id_
                                in_service[j] = 1
                                server_assignment[j] = id_

                                # add a departure event to the event_calendar
                                hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,j))
                                break

                        # log service and departure events
                        event_log.append((ts_, 's', id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))

                        # update wait_time_tracker, wait_time = current time - arrival time
                        wait_tr[classes_[id_]].append(ts_ - sim_arrival_times[id_])
                        # update los_tracker, los = current time + service time - arrival time
                        los_tr[classes_[id_]].append(ts_ + service_times[id_] - sim_arrival_times[id_])
                        # temp_friends[id_] = []

                    # if there is no room on servers
                    else:
                        # temp_friends[id_] = [str(q[1])+"_"+str(r) for q in list(queue)]

                        # join the queue
                        hq.heappush(queue,(ts_, id_))

                        # log queueing event
                        event_log.append((ts_, 'q', id_, interv_[id_], classes_[id_]))

                        # update queue_tracker - add 1 to the class in which the customer belongs to
                        queue_tr[classes_[id_]].append((ts_,queue_tr[classes_[id_]][-1][1] + 1))

                        # update queue_tracker for all other classes, Nq stays the same
                        for class_ in self.classes_:
                            if classes_[id_]!=class_:
                                queue_tr[class_].append((ts_, queue_tr[class_][-1][1]))


                # departure event happens
                else:
                    in_service[server_] = 0  # free the server
                    # server_assignment[server_] = 0
                    # update nis_tracker - subtract 1 to the class in which the customer belongs to
                    nis_tr[classes_[id_]].append((ts_, nis_tr[classes_[id_]][-1][1] - 1))

                    # update nis_tracker for all other classes, NIS stays the same
                    for class_ in self.classes_:
                        if classes_[id_] != class_:
                            nis_tr[class_].append((ts_, nis_tr[class_][-1][1]))

                    # if there is still customer in the queue
                    if len(list(queue)) > 0:
                        # take a customer from the queue
                        _ , id_ = hq.heappop(queue)

                        # log service and departure events
                        event_log.append((ts_,'s',id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))

                        # the server becomes busy again, assign a customer with id_ to the server
                        in_service[server_] = 1
                        server_assignment[server_] = id_

                        # add a departure event to the event_calendar
                        hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,server_))

                        # update wait_time_tracker, los_tracker, queue_tracker (subtract 1)
                        wait_tr[classes_[id_]].append(ts_ - sim_arrival_times[id_])
                        los_tr[classes_[id_]].append(ts_ + service_times[id_] - sim_arrival_times[id_])
                        queue_tr[classes_[id_]].append((ts_, queue_tr[classes_[id_]][-1][1] - 1))

                        # update the queue_tracker for all other classes, Nq stays the same
                        for class_ in self.classes_:
                            if classes_[id_]!=class_:
                                queue_tr[class_].append((ts_, queue_tr[class_][-1][1]))

            # add the event_log to "data", and append trackers for each run to overall trackers
            self.data.append(event_log)
            self.wait_time_tracker.append(wait_tr)
            self.los_tracker.append(los_tr)
            self.nis_tracker.append(nis_tr)
            self.queue_tracker.append(queue_tr)
            # self.trackers.append((self.los_tracker,self.queue_tracker))
            # self.friends.append(temp_friends)

        print('Done simulating...')

    def simulate_q_no_track(self, customers, runs, system_type=1):

        np.random.seed(3)  # set random seed

        for r in range(runs):
            print('running simulation run #: ' + str(r + 1))

            t_ = 0  # time starts from zero
            sim_arrival_times = []
            service_times = []
            classes_ = []
            interv_ = []

            for c in range(customers):
                print('Customer: '+str(c))
                # simulate next arrival
                # next arrival time: t_ + inter-arrival_time
                sim_arrival_times.append(
                    t_ + Distribution(dist_type=DistributionType.exponential, rate=self.lambda_).sample())

                # move forward the timestamp
                t_ = sim_arrival_times[len(sim_arrival_times)-1]

                # sampling the class of arriving patient
                c_ = np.random.choice(self.classes_, p=self.probs)
                classes_.append(c_)

                # sampling whether intervention or not
                interv_.append(np.random.choice(np.array([0,1]),
                                                p = np.array([1-self.probs_speedup[c_], self.probs_speedup[c_]])))
                if interv_[len(interv_)-1]==0:
                    service_times.append(Distribution(dist_type=DistributionType.exponential, rate=self.mus[c_]).sample())
                else:
                    service_times.append(
                        Distribution(dist_type=DistributionType.exponential, rate=self.mus_speedup[c_]).sample())

            # system_type (default 1) --> system_type 1 = M/G/1; system_type 2 = G/G/1 (appointment)
            if system_type == 2:
                # next arrival time: a' = a + noise, noise here is punctuality
                # draw a punctuality time from laplace distribution
                for i, arrival_time in enumerate(sim_arrival_times):
                    done = False
                    while not done:
                        punctuality = Distribution(dist_type=DistributionType.laplace, location=self.laplace_params[0],
                                                   scale=self.laplace_params[1]).sample()
                        actual_arrival_time = arrival_time + punctuality
                        if actual_arrival_time >= 0:
                            sim_arrival_times[i] = actual_arrival_time
                            done = True

            event_log = []
            #queue_tr = [[(0, 0)] for c in self.classes_]  # [[(timestamp, Nq), (timestamp, Nq), ...], ...]
            #nis_tr = [[(0, 0)] for c in self.classes_]  # [[(timestamp, NIS), (timestamp, NIS), ...], ...]
            #los_tr = [[] for c in self.classes_]  # [[timestamp, timestamp, ...], ...]
            #wait_tr = [[] for c in self.classes_]  # [[timestamp, timestamp, ...], ...]
            # four types of events: arrival, departure = 'a', 'd' queue and service (queue start and service start)
            # every tuple is (timestamp, event_type, customer id, server_id)
            event_calendar = [(a, 'a', i, -1) for i,a in enumerate(sim_arrival_times)]
            hq.heapify(event_calendar)

            queue = []  # [(timestamp, customer_id), (timestamp, customer_id), ...]
            hq.heapify(queue)
            # heap is ordered by timestamp - every element is (timestamp, station)
            # need to manage server assignment
            in_service = [0 for s in range(self.servers)]  # 0 = not in service; 1 = in service
            server_assignment = [0 for s in range(self.servers)]
            # temp_friends = {}

            # keep going if there are still events waiting to occur
            while len(list(event_calendar))>0:
                # take an event from the event_calendar
                print(list(event_calendar)[0])
                ts_, event_, id_, server_ = hq.heappop(event_calendar)

                # arrival event happens, need to check if servers are available
                if event_ == 'a':
                    # log arrival event
                    event_log.append((ts_, 'a', id_, interv_[id_], classes_[id_]))

                    # update nis_tracker - add 1 to the class in which the customer belongs to
                    #nis_tr[classes_[id_]].append((ts_, nis_tr[classes_[id_]][-1][1] + 1))

                    # update nis_tracker for all other classes, NIS stays the same
                    #for class_ in self.classes_:
                    #    if classes_[id_] != class_:
                    #        nis_tr[class_].append((ts_, nis_tr[class_][-1][1]))

                    # if there is a room in service
                    if sum(in_service) < self.servers:
                        for j,s in enumerate(in_service):
                            # find the first available server
                            if s == 0:
                                # set the jth server to be busy, serving customer with id_
                                in_service[j] = 1
                                server_assignment[j] = id_

                                # add a departure event to the event_calendar
                                hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,j))
                                break

                        # log service and departure events
                        event_log.append((ts_, 's', id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))

                        # update wait_time_tracker, wait_time = current time - arrival time
                        #wait_tr[classes_[id_]].append(ts_ - sim_arrival_times[id_])
                        # update los_tracker, los = current time + service time - arrival time
                        #los_tr[classes_[id_]].append(ts_ + service_times[id_] - sim_arrival_times[id_])
                        # temp_friends[id_] = []

                    # if there is no room on servers
                    else:
                        # temp_friends[id_] = [str(q[1])+"_"+str(r) for q in list(queue)]

                        # join the queue
                        hq.heappush(queue,(ts_, id_))

                        # log queueing event
                        event_log.append((ts_, 'q', id_, interv_[id_], classes_[id_]))

                        # update queue_tracker - add 1 to the class in which the customer belongs to
                        #queue_tr[classes_[id_]].append((ts_,queue_tr[classes_[id_]][-1][1] + 1))

                        # update queue_tracker for all other classes, Nq stays the same
                        #for class_ in self.classes_:
                        #    if classes_[id_]!=class_:
                        #        queue_tr[class_].append((ts_, queue_tr[class_][-1][1]))


                # departure event happens
                else:
                    in_service[server_] = 0  # free the server
                    # server_assignment[server_] = 0
                    # update nis_tracker - subtract 1 to the class in which the customer belongs to
                    #nis_tr[classes_[id_]].append((ts_, nis_tr[classes_[id_]][-1][1] - 1))

                    # update nis_tracker for all other classes, NIS stays the same
                    #for class_ in self.classes_:
                    #    if classes_[id_] != class_:
                    #        nis_tr[class_].append((ts_, nis_tr[class_][-1][1]))

                    # if there is still customer in the queue
                    if len(list(queue)) > 0:
                        # take a customer from the queue
                        _ , id_ = hq.heappop(queue)

                        # log service and departure events
                        event_log.append((ts_,'s',id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))

                        # the server becomes busy again, assign a customer with id_ to the server
                        in_service[server_] = 1
                        server_assignment[server_] = id_

                        # add a departure event to the event_calendar
                        hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,server_))

                        # update wait_time_tracker, los_tracker, queue_tracker (subtract 1)
                        #wait_tr[classes_[id_]].append(ts_ - sim_arrival_times[id_])
                        #los_tr[classes_[id_]].append(ts_ + service_times[id_] - sim_arrival_times[id_])
                        #queue_tr[classes_[id_]].append((ts_, queue_tr[classes_[id_]][-1][1] - 1))

                        # update the queue_tracker for all other classes, Nq stays the same
                        #for class_ in self.classes_:
                        #    if classes_[id_]!=class_:
                        #        queue_tr[class_].append((ts_, queue_tr[class_][-1][1]))

            # add the event_log to "data", and append trackers for each run to overall trackers
            self.data.append(event_log)
            #self.wait_time_tracker.append(wait_tr)
            #self.los_tracker.append(los_tr)
            #self.nis_tracker.append(nis_tr)
            #self.queue_tracker.append(queue_tr)
            # self.trackers.append((self.los_tracker,self.queue_tracker))
            # self.friends.append(temp_friends)

        print('Done simulating...')


    def generate_data(self, **kwargs):
        # generating data for intervention experiments
        sla_q = kwargs.get('sla_', 0.9)  # service level agreement
        mu_2 = kwargs.get('mu_2', 2.2)
        p_speed = kwargs.get('p_speed', 0.5)
        lam_ = kwargs.get('lam_', 1)
        quant_flag = kwargs.get('quant_flag',True)
        write_file = kwargs.get('write_file', True)


        offset = 0.0  # time at the end of last run
        self.avg_sla_value = 0
        avg_sla_q = 0

        # iterate through each event log (simulation run) in simulation data
        for j,e_l in enumerate(self.data):
            print("Run #"+str(j+1))

            # creating a data-frame to manage the event logs
            # one per simulation run - we will later want to compare interventions
            df = pd.DataFrame(e_l, columns=['timestamp', 'event_type', 'id', 'A', 'C'])
            # two things: we both want plots to see if the simulator makes sense, and create synthetic data
            # print(df.head(5))
            # order by id and timestamp there may be tie between a and q - we don't care
            df.sort_values(by=['id','timestamp'], inplace=True)
            df.reset_index(drop=True,inplace=True)


            # add additional columns to the DataFrame
            df['elapsed'] = 0.0  # time elapsed since customer's arrival
            df['arrival_time'] = 0.0
            df['id_run'] = ""
            df['S'] = 0.0
            cur_id = df.at[0,'id']
            cur_start = df.at[0,'timestamp']
            # df['FriendsID'] = " "
            # df['nFriends'] = 0
            # temp_friends = self.friends[j]

            # go through each event in the DataFrame
            for i in range(len(df)):
                df.at[i,'id_run'] = str(df.at[i,'id'])+"_"+str(j)

                # if the event corresponds to the current customer
                if cur_id == df.at[i,'id']:
                    df.at[i, 'arrival_time'] = cur_start + offset
                    df.at[i,'elapsed'] = df.at[i,'timestamp'] - cur_start
                    # departure event:
                    if df.at[i,'event_type']=='d':
                        df.at[i, 'S'] = df.at[i,'timestamp']-df.at[i-1,'timestamp']
                    #print(df.at[i,'event_type'])
                    #input("Press Enter to continue...")

                # if the event does not correspond to the current customer, events for the next customer starts
                else:

                    cur_id = df.at[i, 'id']  # set current customer to the customer for the event
                    cur_start = df.at[i,'timestamp'].copy()  # advance the current start time to the time of event
                    df.at[i,'arrival_time'] = cur_start + offset
                # df.at[i,'FriendsID'] = " ".join(map(str, temp_friends[df.at[i,'id']]))
                # df.at[i,'nFriends'] = len(temp_friends[df.at[i, 'id']])

            offset = offset + max(df['timestamp']) # the next simulation run starts at the offset time
            print('Average LOS per run: ')
            print(np.mean(df[df.event_type == 'd']['elapsed']))

            #df['SLA'] = 0
            #if quant_flag:
            #    sla_ = np.quantile(df[df.event_type == 'd']['elapsed'], q=sla_q)  # np.mean(df['elapsed'])
            #else:
            #    sla_ = sla_q[j]
            #self.sla_levels.append(sla_)

            #for i in range(len(df)):
            #    if df.at[i, 'elapsed'] > sla_:
            #       df.at[i, 'SLA'] = 1
            # print("Sla level:")
            # print(sum(df[df.event_type == 'd']['SLA']) / len(df[df.event_type == 'd']))

            # generate csv files
            if write_file:

                # save generated data in a folder in the current working directory
                cwd = os.getcwd() # get current working directory
                # single type of customers
                folder = ""
                if len(self.mus) == 1 and False:
                    folder = "Lambda{}Mu{}ProbIntervention{}MuSpeedup{}".format(self.lambda_, self.mus[0],
                                                                                self.probs_speedup[0],
                                                                                self.mus_speedup[0])
                # two types of customers
                elif len(self.mus) == 2 and False:
                    folder = "Lambda{}Mu1{}Mu2{}p1{}ProbIntervention{}Mu1Speedup{}Mu2Speedup{}".format(self.lambda_,
                                                                                                   self.mus[0],
                                                                                                   self.mus[1],
                                                                                                   self.probs[0],
                                                                                                   self.probs_speedup[
                                                                                                       0],
                                                                                                   self.mus_speedup[0],
                                                                                                   self.mus_speedup[1])
                # todo: more than 2 types of customers?

                directory = os.path.join(cwd, folder)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # generate file: 1) Queue/Waiting and System/Time

                header_ = True if j == 0 else False
                mode_ = 'w' if j == 0 else 'a'
                # file_1: Queue/Waiting and System/Time
                filename = "data_WIQ_TIS"
                save_path = os.path.join(directory, filename+".csv")
                #df[df.event_type == 'd'].loc[:,
                #['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'S', 'elapsed']].to_csv(save_path,
                 #                                                                                  mode=mode_,
                  #                                                                                 index=False,
                  #                                                                                 header=header_)
                # wait time = service start time - arrival time
                #df[df.event_type == 's'].loc[:,
                #['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'S', 'elapsed']].to_csv(save_path,
                #                                                                                   mode='a',
                #                                                                                   index=False,
                #                                                                                   header=False)
                # "intervention_data.csv" generator
                if j == 0:
                    # df[df.event_type=='d'].loc[:,['id_run', 'arrival_time', 'event_type','C', 'A', 'elapsed']].to_csv('intervention_data.csv', index=False, header=True)
                    # df[df.event_type == 'd'].loc[:,['id_run', 'arrival_time', 'timestamp', 'event_type','C', 'A', 'FriendsID','nFriends','elapsed', 'SLA']].to_csv('intervention_data.csv', index=False, header=True)
                    df[df.event_type == 'd'].loc[:,
                    ['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'S', 'elapsed']].to_csv(
                        'datafiles\\intervention_data_'+str(lam_)+'_'+str(mu_2)+'_'+str(p_speed)+'.csv', index=False, header=True)

                else:
                    # df[df.event_type=='d'].loc[:,['id_run', 'arrival_time', 'event_type','C', 'A', 'elapsed']].to_csv('intervention_data.csv', mode='a', index= False, header=False)
                    # df[df.event_type == 'd'].loc[:,['id_run', 'arrival_time', 'timestamp','event_type','C', 'A', 'FriendsID','nFriends', 'elapsed','SLA']].to_csv('intervention_data.csv', mode='a', index= False, header=False)
                    df[df.event_type == 'd'].loc[:,
                    ['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'S', 'elapsed']].to_csv(
                        'datafiles\\intervention_data_'+str(lam_)+'_'+str(mu_2)+'_'+str(p_speed)+'.csv', mode='a', index=False, header=False)


        # generate files: 2) Queue/Number, 3) System/Number
        if write_file and False:
            # file_2: Queue/Number
            filename = "data_NIQ"
            save_path = os.path.join(directory, filename + ".csv")
            for r, queue_tr in enumerate(self.queue_tracker):
                df_niq = pd.DataFrame(columns=['run', 'timestamp', 'class_id', 'Number_in_Queue'])
                offset = 0
                for class_ in self.classes_:
                    for i, queue in enumerate(queue_tr[class_]):
                        df_niq.loc[i + offset] = [r+1, queue[0], class_, queue[1]]
                        offset += len(queue_tr[class_])

                df_niq.sort_values(by=['timestamp'], inplace=True)  # order by timestamp
                df_niq.reset_index(drop=True, inplace=True)

                if r == 0:
                    df_niq.to_csv(save_path, index=False, header=True)
                else:
                    df_niq.to_csv(save_path, mode='a', index=False, header=False)

            # file_3: System/Number
            filename = "data_NIS"
            save_path = os.path.join(directory, filename + ".csv")
            for r, nis_tr in enumerate(self.nis_tracker):
                df_nis = pd.DataFrame(columns=['run', 'timestamp', 'class_id', 'Number_in_System'])
                offset = 0
                for class_ in self.classes_:
                    for i, system in enumerate(nis_tr[class_]):
                        df_nis.loc[i + offset] = [r+1, system[0], class_, system[1]]
                        offset += len(nis_tr[class_])

                df_nis.sort_values(by=['timestamp'], inplace=True)  # order by timestamp
                df_nis.reset_index(drop=True, inplace=True)

                if r == 0:
                    df_nis.to_csv(save_path,index=False,header=True)
                else:
                    df_nis.to_csv(save_path, mode='a', index=False, header=False)

        # print("Average SLA value: "+str(np.mean(self.sla_levels)))
    def generate_data_compact(self, **kwargs):
        # generating data for intervention experiments
        mu_2 = kwargs.get('mu_2', 2.2)
        p_speed = kwargs.get('p_speed', 0.5)
        lam_ = kwargs.get('lam_', 1)
        write_file = kwargs.get('write_file', True)
        offset = 0.0  # time at the end of last run
        # iterate through each event log (simulation run) in simulation data
        for j,e_l in enumerate(self.data):
            print("Run #"+str(j+1))

            # creating a data-frame to manage the event logs
            # one per simulation run - we will later want to compare interventions
            df = pd.DataFrame(e_l, columns=['timestamp', 'event_type', 'id', 'A', 'C'])
            # two things: we both want plots to see if the simulator makes sense, and create synthetic data
            # print(df.head(5))
            # order by id and timestamp there may be tie between a and q - we don't care
            df.sort_values(by=['id','timestamp'], inplace=True)
            df.reset_index(drop=True,inplace=True)


            # add additional columns to the DataFrame
            df['elapsed'] = 0.0  # time elapsed since customer's arrival
            df['arrival_time'] = 0.0
            df['id_run'] = ""
            df['S'] = 0.0
            cur_id = df.at[0,'id']
            cur_start = df.at[0,'timestamp']
            # df['FriendsID'] = " "
            # df['nFriends'] = 0
            # temp_friends = self.friends[j]

            # go through each event in the DataFrame
            for i in range(len(df)):
                df.at[i,'id_run'] = str(df.at[i,'id'])+"_"+str(j)

                # if the event corresponds to the current customer
                if cur_id == df.at[i,'id']:
                    df.at[i, 'arrival_time'] = cur_start + offset
                    df.at[i,'elapsed'] = df.at[i,'timestamp'] - cur_start
                    # departure event:
                    if df.at[i,'event_type']=='d':
                        df.at[i, 'S'] = df.at[i,'timestamp']-df.at[i-1,'timestamp']
                    #print(df.at[i,'event_type'])
                    #input("Press Enter to continue...")

                # if the event does not correspond to the current customer, events for the next customer starts
                else:

                    cur_id = df.at[i, 'id']  # set current customer to the customer for the event
                    cur_start = df.at[i,'timestamp'].copy()  # advance the current start time to the time of event
                    df.at[i,'arrival_time'] = cur_start + offset
                # df.at[i,'FriendsID'] = " ".join(map(str, temp_friends[df.at[i,'id']]))
                # df.at[i,'nFriends'] = len(temp_friends[df.at[i, 'id']])

            offset = offset + max(df['timestamp']) # the next simulation run starts at the offset time
            print('Average LOS per run: ')
            print(np.mean(df[df.event_type == 'd']['elapsed']))

            #df['SLA'] = 0
            #if quant_flag:
            #    sla_ = np.quantile(df[df.event_type == 'd']['elapsed'], q=sla_q)  # np.mean(df['elapsed'])
            #else:
            #    sla_ = sla_q[j]
            #self.sla_levels.append(sla_)

            #for i in range(len(df)):
            #    if df.at[i, 'elapsed'] > sla_:
            #       df.at[i, 'SLA'] = 1
            # print("Sla level:")
            # print(sum(df[df.event_type == 'd']['SLA']) / len(df[df.event_type == 'd']))

            # generate csv files
            if write_file:

                # save generated data in a folder in the current working directory
                cwd = os.getcwd() # get current working directory
                # single type of customers
                folder = ""


                directory = os.path.join(cwd, folder)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # generate file: 1) Queue/Waiting and System/Time

                header_ = True if j == 0 else False
                mode_ = 'w' if j == 0 else 'a'
                # file_1: Queue/Waiting and System/Time

                if j == 0:
                    # df[df.event_type=='d'].loc[:,['id_run', 'arrival_time', 'event_type','C', 'A', 'elapsed']].to_csv('intervention_data.csv', index=False, header=True)
                    # df[df.event_type == 'd'].loc[:,['id_run', 'arrival_time', 'timestamp', 'event_type','C', 'A', 'FriendsID','nFriends','elapsed', 'SLA']].to_csv('intervention_data.csv', index=False, header=True)
                    df[df.event_type == 'd'].loc[:,
                    ['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'S', 'elapsed']].to_csv(
                        'datafiles\\intervention_data_'+str(lam_)+'_'+str(mu_2)+'_'+str(p_speed)+'.csv', index=False, header=True)

                else:
                    # df[df.event_type=='d'].loc[:,['id_run', 'arrival_time', 'event_type','C', 'A', 'elapsed']].to_csv('intervention_data.csv', mode='a', index= False, header=False)
                    # df[df.event_type == 'd'].loc[:,['id_run', 'arrival_time', 'timestamp','event_type','C', 'A', 'FriendsID','nFriends', 'elapsed','SLA']].to_csv('intervention_data.csv', mode='a', index= False, header=False)
                    df[df.event_type == 'd'].loc[:,
                    ['id_run', 'arrival_time', 'timestamp', 'event_type', 'C', 'A', 'S', 'elapsed']].to_csv(
                        'datafiles\\intervention_data_'+str(lam_)+'_'+str(mu_2)+'_'+str(p_speed)+'.csv', mode='a', index=False, header=False)




    def performance_los(self):

        run_avg_los = 0
        run_avg_los_class = [0 for c in self.classes_]

        for run, queue in enumerate(self.queue_tracker):
            print("Run #" + str(run+1))
            total_q_list = [sum([queue[c][i][1] for c in range(len(self.classes_))]) for i in range(len(queue[0]))]
            print("Max QL is: " + str(max(total_q_list)))

            #if run == 0:
            #    pw.plot_K_graphs([total_q_list], ['total_queue_length'], 'total_q over time', 'queue_len', ['b'], False,
            #                     0, [], [])

        for run, los in enumerate(self.los_tracker):
            print("Run #" + str(run + 1))
            total_los_run = []
            for j, c in enumerate(self.classes_):
                los_tr = los[c]
                if len(los_tr) > 0:
                    print("LOS for class " + str(c) + ":")
                    los_c = np.mean([los_tr[i] for i in range(len(los_tr))])
                    print(los_c)
                    total_los_run.extend(los_tr)
                    run_avg_los_class[c] += los_c
            if len(total_los_run) > 0:
                print("LOS for all classes: ")
                avg_run_los = np.mean(total_los_run)
                print(avg_run_los)
                run_avg_los += avg_run_los
        print("LOS across runs: " + str(run_avg_los / len(self.los_tracker)))
        for c in self.classes_:
#<<<<<<< HEAD
#            print("LOS per class "+str(c)+": " + str(run_avg_los_class[c]/len(self.trackers)))

#points_ = 20
#runs_ = 1
#mus_speedup_list = [11] #np.linspace(1.1,11,points_)
#sla_list = [3 for i in range(points_)]
#customers_ = 10000
#sla_results =[]
#todo: write different intervention files per value. Run TMLE and plot. Fix sla at 4.
#for j,mu_2 in enumerate(mus_speedup_list):
 #   print('mu 2 is: ' +str(mu_2))
 #   q_ = multi_class_single_station_fcfs(lambda_ = 1, classes = [0], probs = [1.0],
 #                                        mus = [1.1], prob_speedup=[0.3], mus_speedup=[mu_2],
 #                                        servers = 1)

#    q_2 = multi_class_single_station_fcfs(lambda_ = 1, classes = [0], probs = [1.0],
#                                         mus = [1.1], prob_speedup=[0.7], mus_speedup=[mu_2],
#                                         servers = 1)


#    q_.simulate_q(customers = customers_, runs = runs_)

#    sla_results.append(q_.generate_data(sla_ = 0.9, quant_flag=True, write_file = False, filename = 'intervention_'+str(j)))
#=======
            print("LOS per class " + str(c) + ": " + str(run_avg_los_class[c] / len(self.los_tracker)))

if __name__ == "__main__":
    # q_ = multi_class_single_station_fcfs(lambda_ = 1, classes = [0], probs = [1.0],
    #                                      mus = [1.1], prob_speedup=[0.5], mus_speedup=[11],
    #                                      servers = 1)
    #
    # q_2 = multi_class_single_station_fcfs(lambda_ = 1, classes = [0], probs = [1.0],
    #                                      mus = [1.1], prob_speedup=[1.0], mus_speedup=[11],
    #                                      servers = 1)
    #
    #
    # # q_.simulate_q(customers = 100000, runs = 1)
    # q_.simulate_q(customers = 10000, runs = 10)
    # q_.generate_data(sla_ = 0.9, quant_flag=True, write_file = False)
    #
    # q_.performance_los()
    #
    # #fc.calculate_friends("intervention_data.csv", window_ = 5)
    #
    #
    # # q_2.simulate_q(customers = 100000, runs = 1)
    # q_2.simulate_q(customers = 10000, runs = 10)
    #
    # q_2.generate_data(sla_ = q_.sla_levels, quant_flag=False, write_file = False)
    # q_2.performance_los()
    #lambda_ = 0.9
    lam_list = [0.9]
    mu_1 = 1
    #mu_list = [i*mu_1 for i in range(2,11)]
    mu_list = [2]

    customers_ = 100000

    #obs_prob_list = [float(i/10) for i in range(0,11)]
    obs_prob_list = [0.2]
    for lambda_ in lam_list:
        for mu_2 in mu_list:
            for p_speed in obs_prob_list:
                print("Running mu_2 = ", mu_2, "speedup = ", p_speed)
                q_3 = multi_class_single_station_fcfs(lambda_=lambda_, classes=[0, 1], probs=[0.1, 0.9],
                                                       mus=[1, 1.5], prob_speedup=[0.9, 0.1],
                                                       mus_speedup=[2, 2],
                                                       servers=1, laplace_params=[0, 0.5], lognorm_params = [[] , []])
                q_3.simulate_q_no_track(customers=customers_, runs=1, system_type=1)
                #q_3.performance_los()
                q_3.generate_data_compact(write_file=True,
                                  mu_2 = mu_2, p_speed = p_speed, lam_ = lambda_)



        #q_4 = multi_class_single_station_fcfs(lambda_=1, classes=[0], probs=[1.0],
         #                                    mus=[1.1], prob_speedup=[0.5], mus_speedup=[mu_],
          #                                   servers=1, laplace_params=[0, 0.5])
        #q_4.simulate_q(customers=10000, runs=1, system_type=1)
        #q_4.performance_los()

        #q_4.generate_data(sla_=q_3.sla_levels, quant_flag=False, write_file=False)