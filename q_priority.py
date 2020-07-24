# The next step would be to create a new Python Class for Priority M/G/1 multi-class queues.
# For now, assume that we have a list of classes 1,…,K and the priority respects the order (1 is highest priority, K is lowest).
# Priorities are strict (not randomized) so if type i is in queue it will always get priority over i+1, i+2, ...

# Comments:
# 1. Separate queueing systems with FCFS queue discipline with priority
# 2. Use a different python file to generate_data --> otherwise, would be repetitive for 2 classes
# 3. Create getter functions in case need to get the data members in another python file
# 4. Change function name "simulate_q" for FCFS queue to be "simulate_q_FCFS"

# Implementations:
# 1. FCFS class --> class object creation, simulate_q_fsfc
# 2. Priority class --> class object creation, simulate_q_priority
# 3. strictly generate_data function (doesn't include sla_ level calculations)
# 4. plotting --> generate plots, calculate mean & 90th percentiles for tis, wiq, niq, wiq

from resource import *
import heapq as hq
import numpy as np

class multi_class_single_station_priority:
    # defining the queueing system using given parameters
    def __init__(self, **kwargs):

        # initialize parameters
        self.lambda_ = kwargs.get('lambda_', 1)  # arrival rate
        self.classes_ = kwargs.get('classes', [0])  # class types

        self.probs = kwargs.get('probs', [1])  # probability of arrival for each class

        self.mus = kwargs.get('mus',[1])  # service rate without intervention
        self.probs_speedup = kwargs.get('prob_speedup', [0]*len(self.classes_))  # probability of speedup

        self.mus_speedup = kwargs.get('mus_speedup', self.mus)  # service rate with intervention
        self.servers = kwargs.get('servers', 1)  # number of servers

        self.priority = kwargs.get('priority', [0])  # priority assignment, smaller number means higher priority

        # initialize trackers with relevant statistics, assume all start empty
        self.data = []  # event logs
        self.wait_time_tracker = []  # [[[timestamp, timestamp, ...], ...]]
        self.los_tracker = []  # [[[timestamp, timestamp, ...], ...]]
        self.queue_tracker = []  # [[[(timestamp, Nq), (timestamp, Nq), ...], ...]]
        self.nis_tracker = []  # [[[(timestamp, NIS), (timestamp, NIS), ...], ...]]


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


    def simulate_priority_q(self, customers, runs):

        np.random.seed(3)  # set random seed

        for r in range(runs):
            print('running simulation run #: ' + str(r + 1))

            t_ = 0  # time starts from zero
            sim_arrival_times = []
            service_times = []
            classes_ = []
            interv_ = []

            # system_type: 1 = M/G/1, 2 = G/G/1 (appointment)
            for c in range(customers):
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

            event_log = []
            queue_tr = [[(0, 0)] for c in self.classes_]  # [[(timestamp, Nq), (timestamp, Nq), ...], ...]
            nis_tr = [[(0, 0)] for c in self.classes_]  # [[(timestamp, NIS), (timestamp, NIS), ...], ...]
            los_tr = [[] for c in self.classes_]  # [[timestamp, timestamp, ...], ...]
            wait_tr = [[] for c in self.classes_]  # [[timestamp, timestamp, ...], ...]
            # four types of events: arrival, departure = 'a', 'd' queue and service (queue start and service start)
            # every tuple is (timestamp, event_type, customer id, server_id)
            event_calendar = [(a, 'a', i, -1, self.priority[classes_[i]]) for i, a in enumerate(sim_arrival_times)]
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
                ts_, event_, id_, server_, priority_ = hq.heappop(event_calendar)

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
                                hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,j, self.priority[classes_[id_]]))
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
                        hq.heappush(queue,(priority_, ts_, id_))  # smaller number means higher priority

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
                        _, _ , id_ = hq.heappop(queue)

                        # log service and departure events
                        event_log.append((ts_,'s',id_, interv_[id_], classes_[id_]))
                        event_log.append((ts_+service_times[id_], 'd', id_, interv_[id_], classes_[id_]))

                        # the server becomes busy again, assign a customer with id_ to the server
                        in_service[server_] = 1
                        server_assignment[server_] = id_

                        # add a departure event to the event_calendar
                        hq.heappush(event_calendar, (ts_ + service_times[id_], 'd', id_,server_, self.priority[classes_[id_]]))

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

        print('Done simulating...')


if __name__ == "__main__":
    q_priority_1 = multi_class_single_station_priority(lambda_=1, classes=[0], probs=[1.0], mus=[1.1],
                                                       prob_speedup=[0.5], mus_speedup=[11], servers=1,
                                                       priority=[0])
    q_priority_1.simulate_priority_q(customers=100, runs=3)

    q_priority_2 = multi_class_single_station_priority(lambda_=1, classes=[0, 1], probs=[0.5, 0.5], mus = [0.5,2],
                                                       prob_speedup=[0.0,0.0], mus_speedup=[5,2], servers = 2,
                                                       priority=[0, 1])
    q_priority_2.simulate_priority_q(customers=100, runs=3)