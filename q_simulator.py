import pandas as pd
from resource import *
import heapq as hq
import numpy as np
import time

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
        self.context = kwargs.get('context', [])  # number of servers
        self.context_weights = kwargs.get('context_weights', [])
        self.laplace_params = kwargs.get('laplace_params', [0, 0.5])  # location and scale parameters

    def log_update(self,log, customer, event, timestamp, server, class_, nis, niq, nservice, nis_vec, customer_contexts):

        #log  = {'customer':[], 'event':[], 'timestamp':[], 'server':[], 'class':[], 'nis':[], 'niq':[]}
        log['customer'].append(customer)
        log['event'].append(event)
        log['timestamp'].append(timestamp)
        log['server'].append(server)
        log['class'].append(class_)
        log['nis'].append(nis)
        log['niq'].append(niq)
        log['nservice'].append(nservice)
        for j,c in enumerate(self.context):
            log[c].append(customer_contexts[j])
        for c in self.classes_:
            log['nis_class_'+str(c)].append(nis_vec[c])

        return

    def simulate_q(self, customers, runs, system_type=1):

        np.random.seed(3)  # set random seed
        log  = {'customer':[], 'event':[], 'timestamp':[], 'server':[], 'class':[], 'nis':[], 'niq':[], 'nservice':[]}
        for c in self.classes_:
            log['nis_class_'+str(c)] = []
        for c in self.context:
            log[c] = []
        for r in range(runs):
            print('running simulation run #: ' + str(r + 1))

            t_ = 0  # time starts from zero
            sim_arrival_times = []
            service_times = []
            customer_classes_ = []
            customer_contexts = []
            for c in range(customers):
                #if c%1000==0:
                #    print('Customer: '+str(c))
                # simulate next arrival
                # next arrival time: t_ + inter-arrival_time
                sim_arrival_times.append(
                    t_ + Distribution(dist_type=DistributionType.exponential, rate=self.lambda_).sample())

                # move forward the timestamp
                t_ = sim_arrival_times[len(sim_arrival_times)-1]

                # sampling the class of arriving patient
                class_ = np.random.choice(self.classes_, p=self.probs)
                customer_classes_.append(class_)
                service_times.append(
                    Distribution(dist_type=DistributionType.exponential, rate=self.mus[class_]).sample())


                context_vec = []
                for c_ in self.context:
                    context_vec.append(random.choices([0, 1], weights=self.context_weights[c_])[0])
                customer_contexts.append(context_vec)



            event_calendar = [(a, 'a', i, -1) for i,a in enumerate(sim_arrival_times)]
            hq.heapify(event_calendar)
            nis_vec = []
            for c in self.classes_:
                nis_vec.append(0)
            total_nis = 0
            queue = []  # [(timestamp, customer_id), (timestamp, customer_id), ...]
            #hq.heapify(queue)
            # heap is ordered by timestamp - every element is (timestamp, station)
            # need to manage server assignment
            in_service = [0 for s in range(self.servers)]  # 0 = not in service; 1 = in service
            server_assignment = [0 for s in range(self.servers)]
            # temp_friends = {}

            # keep going if there are still events waiting to occur - till last departure
            while len(list(event_calendar))>0:
                # take an event from the event_calendar

                #if len(event_calendar)%10000==0:
                #    print(len(event_calendar))
                ts_, event_, id_, server_ = hq.heappop(event_calendar)

                self.log_update(log, customer= id_, event = event_, timestamp = ts_,
                                server = server_, class_ = customer_classes_[id_],
                                nis = total_nis ,niq = len(queue), nservice=sum(in_service),
                                nis_vec=nis_vec, customer_contexts = customer_contexts[id_])
                # arrival event happens, need to check if servers are available
                if event_ == 'a':
                    total_nis +=1
                    nis_vec[customer_classes_[id_]] += 1
                    # if there is a room in service
                    if sum(in_service) < self.servers:
                        for j,s in enumerate(in_service):
                            # find the first available server
                            if s == 0:
                                # set the jth server to be busy, serving customer with id_
                                in_service[j] = 1
                                server_assignment[j] = id_

                                self.log_update(log, customer=id_, event='s', timestamp=ts_, server=j,
                                                class_=customer_classes_[id_], nis=total_nis, niq=len(queue), nservice = sum(in_service),
                                                nis_vec= nis_vec, customer_contexts = customer_contexts[id_])
                                # add a departure event to the event_calendar
                                hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,j))
                                break

                        # log service and departure events


                    # if there is no room on servers
                    else:
                        # temp_friends[id_] = [str(q[1])+"_"+str(r) for q in list(queue)]

                        # join the queue
                        #hq.heappush(queue,(ts_, id_))
                        queue.insert(0,(ts_,id_))




                # departure event happens
                else:
                    total_nis -= 1
                    nis_vec[customer_classes_[id_]] -= 1
                    in_service[server_] = 0  # free the server
                    #enqueueign a new guy

                    if len(list(queue)) > 0:
                        # take a customer from the queue
                        _ , id_ = queue.pop(len(queue)-1) #hq.heappop(queue)

                        # log service and departure events
                        #event_log.append((ts_,'s',id_, interv_[id_], classes_[id_]))
                        #event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))

                        # the server becomes busy again, assign a customer with id_ to the server
                        in_service[server_] = 1
                        server_assignment[server_] = id_
                        self.log_update(log, customer=id_, event='s', timestamp=ts_, server=server_,
                                        class_=customer_classes_[id_], nis=total_nis,
                                        niq=len(queue), nservice=sum(in_service), nis_vec=nis_vec, customer_contexts = customer_contexts[id_])

                        # add a departure event to the event_calendar
                        hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,server_))




            df_log = pd.DataFrame.from_dict(log)
            df_log.to_csv('simulation_result_run_'+str(r)+'.csv', index=False)

        print('Done simulating...')



if __name__ == "__main__":

    #lambda_ = 0.9
    start = time.time()
    #print("start time is", start)


    customers_ = 100000
    runs_ = 1
    arrival_rate = 100
    class_proportions = [0.5, 0.3, 0.2] #[1]
    classes_ = [0, 1, 2]
    service_rates = [3, 2, 1] #[2, 4, 1]
    beta = 0.5#5
    offered_load = arrival_rate * (sum((p/m) for p, m in zip(class_proportions, service_rates)))

    print("Offered load:", offered_load)
    num_Servers =int(offered_load+np.sqrt(offered_load)*beta)
    print("Number of servers:", num_Servers)
    print("Utilization:", np.round(offered_load/num_Servers,4))
    context = ['x1', 'x2']
    weight_vector_p1 = [0.8, 0.2]
    context_weights = {}
    for j,c in enumerate(context):
        context_weights[c] = [1-weight_vector_p1[j], weight_vector_p1[j]]

    q = multi_class_single_station_fcfs(lambda_=arrival_rate, classes=classes_, probs=class_proportions,
                                          mus=service_rates, servers=num_Servers, context =context, context_weights=context_weights)#,
                                        #prob_speedup=[0,0,0],
                                          #mus_speedup=[2,2,2],
                                          #laplace_params=[0], lognorm_params=[[], []])

    q.simulate_q(customers=customers_, runs=runs_, system_type=1)
    end = time.time()
    #print("end time is", end)

    print('total duration of '+str(runs_)+' runs is',  end - start)