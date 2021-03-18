from collections import namedtuple, defaultdict, deque
from enum import Enum
import math
import copy
import random
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from scipy.optimize import linear_sum_assignment
import numpy as np

#### read in input ###########
Street = namedtuple("Street", ['start', 'end', 'name', 'duration', 'id'])
Car = namedtuple("Car", ["id", "path"])
Intersection = namedtuple("Intersection", "incoming")
D, I, S, V, F = map(int, input().split())
streets = []
street_name_to_street_id = dict()
intersections = [Intersection(incoming=defaultdict(int)) for i in range(I)]
for i in range(S):
    s, e, n, d = input().split()
    streets.append(Street(int(s), int(e), n, int(d), i))
    # intersections[int(e)].incoming[i] += 1
    street_name_to_street_id[n] = i
cars = []
for i in range(V):
    inp = input().split()
    path = inp[1:]
    nice_path = [street_name_to_street_id[x] for x in path]
    cars.append(Car(i,nice_path))
################################

for car in cars:
    for street_id in car.path[:-1]:
        intersections[streets[street_id].end].incoming[street_id] += 1

def intersections_of_path(street_ids):
    return [street_ids[0].start] + [streets[sid].end for sid in street_ids]

""" ICar: instantaneous car.
street_of_path = 2 means that the car is on the 2th street in its path.
duration_on_street = 3 means the car has been on that street for 3 seconds.  (-1 means it's crossing the intersection) """
class CarStatus(Enum):
    WAITING = 0
    DRIVING = 1
    DONE = 2

class ICar:
    def __init__(self, car, status, street_of_path, duration_on_street):
        self.car = car
        self.status = status
        self.street_of_path = street_of_path
        self.duration_on_street = duration_on_street
    def __str__(self):
        return '{}, {}, {}, {}'.format(self.car, self.status, self.street_of_path, self.duration_on_street)

""" IWorld: instantaneous world.
"""
Update = namedtuple("Update", ['car_id'])
class IWorld:
    def __init__(self, streets, cars, intersections):
        self.streets = streets
        self.queues = [deque([]) for i in streets]
        self.icars = [ICar(car, status=CarStatus.WAITING, street_of_path=0, duration_on_street=0) for car in cars]
        for icar in self.icars:
            self.queues[icar.car.path[0]].append(icar)
    def update(self, update):
        icar = self.icars[update.car_id]
        if icar.status == CarStatus.WAITING:
            # icar.status = CarStatus.CROSSING
            self.queues[icar.car.path[icar.street_of_path]].popleft()
        # elif icar.status == CarStatus.CROSSING:
            icar.street_of_path += 1
            icar.duration_on_street = 1
            icar.status = CarStatus.DRIVING
            if self.streets[icar.car.path[icar.street_of_path]].duration == 1:
                self.update(update)
        elif icar.status == CarStatus.DRIVING:
            if icar.street_of_path == len(icar.car.path)-1:
                icar.status = CarStatus.DONE
            else:
                icar.status = CarStatus.WAITING
                self.queues[icar.car.path[icar.street_of_path]].append(icar)
        elif icar.status == CarStatus.DONE:
            raise ValueError("done car shouldn't need update")
        else:
            raise ValueError('unrecognized status {}'.format(icar.status))
        return icar.status


class Sim:
    def __init__(self, disvf, streets, cars, intersections):
        D, I, S, V, F = disvf
        self.D = D
        self.I = I
        self.S = S
        self.V = V
        self.F = F
        self.streets = streets
        self.cars = cars
        self.intersections = intersections
        self.iworld = IWorld(streets, cars, intersections)
        self.score_cache = None
    
    def get_greedy_schedule(self, fixed_demands):
        unsatisfied_demands_by_intersection = defaultdict(list)
        all_demands_by_intersection = defaultdict(list)
        duration_per_street = defaultdict(lambda: 1)
        print('making schedule')
        schedule = Schedule(dict(), streets)
        for intersection_id, intersection in enumerate(self.intersections):
            if len(intersection.incoming) == 0:
                continue
            schedule[intersection_id] = IntersectionScheduleGreedy(
                intersection_id,
                [(None, 1) for i in intersection.incoming],
                list(intersection.incoming)
            )
            # to speed this up, this could all be done at build-time in one pass:
            for street_id in fixed_demands:
                if street_id in intersection.incoming:
                    time, duration = fixed_demands[street_id]
                    if schedule[intersection_id].time_is_unset(time):
                        schedule[intersection_id].set(street_id, time, duration)
            schedule[intersection_id].make_cache()
        print('building iworld')
        self.iworld = IWorld(streets, cars, intersections)
        score = 0
        print('beginning sim')
        for d in tqdm(range(self.D)):
            #### TODO: should try to use the latest time of arrival instead of earliest!
            updates = []
            for icar in self.iworld.icars:
                street_id = icar.car.path[icar.street_of_path]
                if icar.status == CarStatus.WAITING:
                    queue = self.iworld.queues[street_id]
                    if icar == queue[0]:
                        if schedule.intersection_of(street_id).street_is_unset(street_id):
                            schedule.intersection_of(street_id).set_soonest(street_id, d, duration_per_street[street_id])
                        if schedule.intersection_of(street_id).is_green(street_id, d):
                            updates.append(Update(icar.car.id))
                        else:
                            street = self.streets[street_id]
                            unsatisfied_demands_by_intersection[street.end].append((street_id, d))
                elif icar.status == CarStatus.DRIVING:
                    icar.duration_on_street += 1
                    if icar.duration_on_street == self.streets[icar.car.path[icar.street_of_path]].duration:
                        updates.append(Update(icar.car.id)) 
                elif icar.status == CarStatus.DONE:
                    continue
                else:
                    raise ValueError('unrecognized car status {}'.format(icar.status))

            for update in updates:
                new_status = self.iworld.update(update)
                if new_status == CarStatus.DONE:
                    score += self.F + self.D - d - 1
                elif new_status == CarStatus.WAITING:
                    icar = self.iworld.icars[update.car_id]
                    street_id = icar.car.path[icar.street_of_path]
                    street = self.streets[street_id]
                    all_demands_by_intersection[street.end].append((street_id, d+1))
        for intersection_id in schedule:
            schedule[intersection_id].fill_in_rest()
            schedule[intersection_id].apply_manual_hacks()
        ##### printing metrics ########
        # timeline = []
        # c = 0
        # for sid,d in schedule[5].schedule:
        #     timeline.append((sid, c))
        #     c += d
        # print(*timeline)
        ###############################
        self.unsatisfied_demands_by_intersection = unsatisfied_demands_by_intersection
        self.all_demands_by_intersection = all_demands_by_intersection
        print(score)
        self.score_cache = score
        return schedule

    def score(self, schedule, debug=False):
        all_demands_by_intersection = defaultdict(list)
        ####### traffic debugging ############
        def debugger(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
        TRAFFIC_THRESH = 2
        intersection_traffic = defaultdict(int)
        street_traffic = defaultdict(int)
        ######################################
        if self.score_cache is not None:
            return self.score_cache
        self.iworld = IWorld(streets, cars, intersections)
        score = 0
        n = 0
        for d in range(self.D):
            updates = []
            for icar in self.iworld.icars:
                if icar.status == CarStatus.WAITING:
                    street_id = icar.car.path[icar.street_of_path]
                    queue = self.iworld.queues[street_id]
                    if icar == queue[0]:
                        if schedule.is_green(street_id, d):
                            updates.append(Update(icar.car.id))
                        if len(queue) >= TRAFFIC_THRESH and debug:
                            street = self.streets[street_id]
                            debugger('intersection {}: street {} ({}). At time {}, backed up w {} cars. t={}'.format(
                                street.end, street_id, street.name, d, len(queue), d%schedule[street.end].period()))
                            intersection_traffic[street.end] += 1
                            street_traffic[street_id] += 1
                elif icar.status == CarStatus.DRIVING:
                    icar.duration_on_street += 1
                    if icar.duration_on_street == self.streets[icar.car.path[icar.street_of_path]].duration:
                        updates.append(Update(icar.car.id))
                elif icar.status == CarStatus.DONE:
                    continue
                else:
                    raise ValueError('unrecognized car status {}'.format(icar.status))

            for update in updates:
                new_status = self.iworld.update(update)
                if new_status == CarStatus.DONE:
                    score += self.F + self.D - d - 1
                    n += 1
                elif new_status == CarStatus.WAITING:
                    icar = self.iworld.icars[update.car_id]
                    street_id = icar.car.path[icar.street_of_path]
                    street = self.streets[street_id]
                    all_demands_by_intersection[street.end].append((street_id, d+1))
        print(n, 'cars,', score, 'score')
        debugger('total seconds lost: {} ({})'.format(sum(intersection_traffic.values()), sum(street_traffic.values())))
        debugger('traffic by intersection:', sorted(intersection_traffic.items(),key=lambda x:x[1]))
        debugger('traffic by street:', sorted(street_traffic.items(),key=lambda x:x[1]))
        self.score_cache = score
        self.all_demands_by_intersection = all_demands_by_intersection
        return score

class IntersectionSchedule:
    """
    We assume that the period is set upon creation and will never change.  This is a decently realistic constraint b/c making changes that
    affect the period will unpredictably affect anything in the schedule that's already set, so there shouldn't be too much use in doing so.
    
    We currently also assume that all the timeslot durations are preset and the only thing that will be changed online are the placement of streets into the slots.
    e.g.: the schedule is preset at [1,1,1,1,2,2,1,1,1,3] and some subset of the slots are up for grabs.
    """
    def __init__(self, id, schedule: [('street_id', 'duration')]):
        self.id = id
        self.schedule = schedule
        self._time_to_index = None
        self._period = None
    def make_cache(self):
        self._time_to_index = []
        for index, (street_id, duration) in enumerate(self.schedule):
            for i in range(duration):
                self._time_to_index.append(index)
        self._cache_period()
    def _cache_period(self):
        self._period = self._calculate_period()
    def _calculate_period(self):
        return sum(duration for street_id, duration in self.schedule)
    def period(self):
        if self._period is not None:
            return self._period
        return self._calculate_period()
    def index_at_time(self, time):
        time = time%self.period()
        if self._time_to_index is not None:
            return self._time_to_index[time]
        else:
            t = 0
            for index, (s_id, duration) in enumerate(self.schedule):
                t += duration
                if t > time:
                    return index
    # def get_street(self, street_id):
    #     t = 0
    #     for s_id, duration in self.schedule:
    #         if s_id == street_id:
    #             return t,duration
    #         t += duration
    def street_id_at_time(self, time):
        return self.schedule[self.index_at_time(time)][0]
        # t = 0
        # for s_id, duration in self.schedule:
        #     t += duration
        #     if t > time:
        #         return s_id
    def is_green(self, street_id, time):
        # THIS IS REALLY SLOW.  CAN BE AMORTIZED O(1)-ish (persisting a pointer), OR O(LOGN) (binary search each time) INSTEAD OF O(N) where N is number of streets on this intersection
        # or amortized O(1) with O(period) space
        return self.street_id_at_time(time) == street_id
    def mutate(self):
        random.shuffle(self.schedule)

class IntersectionScheduleGreedy(IntersectionSchedule):
    def __init__(self, id, schedule: [('street_id', 'duration')], incoming_streets):
        if schedule is None:
            schedule = [(street_id, 1) for street_id in streets]
        self.unused_streets = set(incoming_streets)
        super().__init__(id, schedule)
    def time_is_unset(self, time):
        return self.street_id_at_time(time) is None
    def street_is_unset(self, street_id):
        return street_id in self.unused_streets
    def set(self, street_id, time, duration=1):
        assert self.street_id_at_time(time) is None
        self.schedule[self.index_at_time(time)] = (street_id, duration)
        self.unused_streets.remove(street_id)
    def unsafe_set(self, street_id, time, duration=1):
        # THIS DESTROYS THE GREEDY BOOKKEEPING OF THIS CLASS.
        self.schedule[self.index_at_time(time)] = (street_id, duration)
    def set_soonest(self, street_id, time, duration=1):
        start = self.index_at_time(time)
        for i in range(len(self.schedule)):
            index = (start+i)%len(self.schedule)
            if self.schedule[index][0] is None:
                self.schedule[index] = (street_id, duration)
                self.unused_streets.remove(street_id)
                break
    def fill_in_rest(self):
        iterator = iter(self.unused_streets)
        for i in range(len(self.schedule)):
            if self.schedule[i][0] is None:
                self.schedule[i] = (next(iterator), 1)
    def apply_manual_hacks(self):
        pass

class Schedule(dict):
    def __init__(self, schedule, streets):
        self.streets = streets
        super().__init__(schedule)
    def intersection_of(self, street_id):
        return self[self.streets[street_id].end]
    def is_green(self, street_id, time):
        street = self.streets[street_id]
        intersection_schedule = self[street.end]
        return intersection_schedule.is_green(street_id, time)
    def serialize(self):
        lines = []
        def add_line(line):
            lines.append(line)
        schedule = self
        add_line(len(schedule))
        for inter_id, inter in schedule.items():
            add_line(inter.id)
            add_line(len(inter.schedule))
            assert len(inter.schedule) > 0
            for street_id, duration in inter.schedule:
                add_line(streets[street_id].name + ' ' + str(duration))
        return '\n'.join(str(line) for line in lines)
    def clone(self):
        return copy.deepcopy(self)

##################################################################################
### Copied from my AZsPCs template
###
class Solution:
    def sample_neighbors(self, n: int, temperature):
        return (self.sample_neighbor(temperature) for i in range(n)) 
    def sample_neighbor(self, temperature):
        """temperature is a float from 0 to 1.  1 meaning lots of change, 0 being no change."""
        raise NotImplementedError 
    def heuristic(self):
        # heuristic score to use when evaluating whether a solution is more promising or not to explore
        raise NotImplementedError 
    def score(self):
        # formal score in azspcs for this solution
        raise NotImplementedError 
    def is_feasible(self):
        raise NotImplementedError 
    def is_potentially_feasible(self):
        raise NotImplementedError
    def serialize(self):
        raise NotImplementedError

class MonteCarloBeamSearcher:
    def __init__(self, start, best_of_all_time):
        self.solution = start
        self.best_of_all_time = best_of_all_time
    def go(self, iterations = 1000, population = 10, samples = 20):
        # linear temperature function:
        temperature_function = lambda x: 1 - x/iterations 
        if self.solution is None:
            raise ValueError
        candidates = [self.solution]
        ans = self.solution
        for it in tqdm(range(iterations)):
            if it%3 == 0:
                for cand in candidates:
                    print(cand.score())
                    print(cand.pretty())
            next_candidates = [candidates[0]]
            for cand in candidates:
                next_candidates.extend(cand.sample_neighbors(samples, temperature=temperature_function(it)))
            b = max(*next_candidates, key = lambda x: x.score())
            bscore = b.score()
            if bscore > ans.score():
                print(bscore)
                # print(b)
                # print(b.serialize())
                ans = b
            if bscore > self.best_of_all_time:
                b.save()
                self.best_of_all_time = bscore
            else:
                print(ans.score(), bscore)
            next_candidates.sort(key=lambda x: x.heuristic(), reverse=True)
            candidates = next_candidates[:population]
        return ans
###################################################################################

class DemandBasedTrafficSignalingSolution(Solution):
    def __init__(self, fixed_demands):
        self.sim = Sim([D,I,S,V,F], streets, cars, intersections)
        self.fixed_demands = fixed_demands
        self.schedule = self.sim.get_greedy_schedule(fixed_demands)
    def pretty(self):
        return str(self.fixed_demands)
    def serialize(self):
        return self.schedule.serialize()
    def sample_neighbor(self, temperature):
        return self.sample_neighbor_sampled(temperature)
    def sample_neighbor_sampled(self, temperature):
        assert 0<=temperature<=1
        def get_random_duration():
            return random.choices([1,2,3],weights=[0.5,0.4,0.1],k=1)[0]
        def get_value(lower_bound, upper_bound):
            # inclusive, both ends
            return lower_bound+(upper_bound-lower_bound)*temperature
        new_fixed_demands = {k:v for k,v in self.fixed_demands.items()}
        # maybe delete some existing demands
        if random.random() < 0.5:
            lb, ub = 0, 2
            k = min(int(get_value(lb,ub)), len(new_fixed_demands))
            for i in random.sample(list(new_fixed_demands),k):
                new_fixed_demands.pop(i)
        # maybe move around some existing demands
        if random.random() < 0.5:
            lb, ub = 1, 3
            k = min(int(get_value(lb,ub)), len(new_fixed_demands))
            earlier_delta = int(-10 * temperature)
            later_delta = int(40*temperature)
            for i in random.sample(list(new_fixed_demands), k):
                time, duration = new_fixed_demands[i]
                new_fixed_demands[i] = (time+random.randint(earlier_delta,later_delta), duration)
        # maybe change the durations of some existing demands
        if random.random() < 0.35:
            lb, ub = 1,3
            k = min(int(get_value(lb,ub)), len(new_fixed_demands))
            for i in random.sample(list(new_fixed_demands), k):
                time, duration = new_fixed_demands[i]
                new_fixed_demands[i] = (time, get_random_duration())
        # satisfy some unsatisfied demands
        intersections_to_choose = [(k, len(v)) for k, v in self.sim.unsatisfied_demands_by_intersection.items()]
        pop, weights = zip(*intersections_to_choose)
        intersection_id = random.choices(pop,weights,k=1)[0]
        k = max(1, int(temperature*4))
        demands_to_satisfy = random.sample(self.sim.unsatisfied_demands_by_intersection[intersection_id], k)
        intersection = self.schedule[intersection_id]
        for street_id, time in demands_to_satisfy:
            new_fixed_demands[street_id] = (time%intersection.period(), get_random_duration())
        # done
        print(new_fixed_demands)
        return DemandBasedTrafficSignalingSolution(new_fixed_demands)
    def sample_neighbor_perfect_matching(self, temperature):
        intersections_to_choose = [(23, 100), (63, 100), (83, 103), (0, 103), (3, 106), (24, 109), (82, 114), (53, 145), (11, 170), (16, 194), (45, 217), (28, 227), (12, 251), (4, 472), (10, 553), (8, 605), (5, 876)]
        pop, weights = zip(*intersections_to_choose)
        intersection_id = random.choices(pop,weights,k=1)[0]
        demands_to_satisfy = self.sim.all_demands_by_intersection[intersection_id]
        # make cost matrix
        period = self.schedule[intersection_id].period()
        demands_per_street = {j:[0 for i in range(period)] for j in intersections[intersection_id].incoming}
        for street_id, time in demands_to_satisfy:
            demands_per_street[street_id][time%period] += 1
        # cost_matrix: each row is a different street.  each column is a different time.
        ordered_street_ids = list(demands_per_street)
        assert len(ordered_street_ids) == period
        cost_matrix = np.zeros((len(ordered_street_ids),period))
        for order, street_id in enumerate(ordered_street_ids):
            total_waiting_time = 0
            total_demands = 0
            for t in range(period):
                total_waiting_time += demands_per_street[street_id][t] * (period-t)%period
                total_demands += demands_per_street[street_id][t]
            for t in range(period):
                cost_matrix[order][t] = total_waiting_time
                total_waiting_time += total_demands-demands_per_street[street_id][t]
                total_waiting_time -= (period-1) * demands_per_street[street_id][t]
        # compute min-cost perfect matching
        _, col_ind = linear_sum_assignment(cost_matrix)
        # return new fixed demands
        new_fixed_demands = {k:v for k,v in self.fixed_demands.items()}
        for street_id, t in zip(ordered_street_ids, col_ind):
            new_fixed_demands[street_id] = (t, 1)
        print(new_fixed_demands)
        return DemandBasedTrafficSignalingSolution(new_fixed_demands)

    def score(self):
        return self.sim.score(self.schedule)
    def heuristic(self):
        return self.score()
    def save(self):
        with open('kaggle.out', 'w') as f:
            f.write(self.serialize())
        with open('kaggle_score.out', 'w') as f:
            f.write(str(self.score()))

class ScheduleBasedTrafficSignalingSolution(Solution):
    def __init__(self, schedule):
        self.schedule = schedule
        self.sim = Sim([D,I,S,V,F], streets, cars, intersections)
        # self.fixed_demands = fixed_demands
        self.sim.score(schedule)
    def pretty(self):
        return str('shrug')
    def serialize(self):
        return self.schedule.serialize()
    def sample_neighbor(self, temperature):
        intersections_to_choose = [(23, 100), (63, 100), (83, 103), (0, 103), (3, 106), (24, 109), (82, 114), (53, 145), (11, 170), (16, 194), (45, 217), (28, 227), (12, 251), (4, 472), (10, 553), (8, 605), (5, 876)]
        pop, weights = zip(*intersections_to_choose)
        intersection_id = random.choices(pop,weights,k=1)[0]
        demands_to_satisfy = self.sim.all_demands_by_intersection[intersection_id]
        # make cost matrix
        period = self.schedule[intersection_id].period()
        demands_per_street = {j:[0 for i in range(period)] for j in intersections[intersection_id].incoming}
        for street_id, time in demands_to_satisfy:
            demands_per_street[street_id][time%period] += 1
        # cost_matrix: each row is a different street.  each column is a different time.
        ordered_street_ids = list(demands_per_street)
        assert len(ordered_street_ids) == period
        cost_matrix = np.zeros((len(ordered_street_ids),period))
        for order, street_id in enumerate(ordered_street_ids):
            total_waiting_time = 0
            total_demands = 0
            for t in range(period):
                total_waiting_time += demands_per_street[street_id][t] * (period-t)%period
                total_demands += demands_per_street[street_id][t]
            for t in range(period):
                cost_matrix[order][t] = total_waiting_time
                total_waiting_time += total_demands-demands_per_street[street_id][t]
                total_waiting_time -= (period-1) * demands_per_street[street_id][t]
        # compute min-cost perfect matching
        _, col_ind = linear_sum_assignment(cost_matrix)
        # write to new schedule
        new_schedule = copy.deepcopy(self.schedule)
        # new_fixed_demands = {k:v for k,v in self.fixed_demands.items()}
        for street_id, t in zip(ordered_street_ids, col_ind):
            new_schedule[intersection_id].unsafe_set(street_id, t, duration=1)
        return ScheduleBasedTrafficSignalingSolution(new_schedule)

    def score(self):
        return self.sim.score(self.schedule)
    def heuristic(self):
        return self.score()
    def save(self):
        with open('kaggle.out', 'w') as f:
            f.write(self.serialize())
        with open('kaggle_score.out', 'w') as f:
            f.write(str(self.score()))

def main():
    try:
        with open('kaggle_score.out', 'r') as f:
            best = int(f.readline())
    except FileNotFoundError:
        with open('kaggle_score.out', 'w') as f:
            f.write('0')
        best = 0

    # sim = Sim([D,I,S,V,F], streets, cars, intersections)
    # schedule = sim.get_greedy_schedule(dict())
    # sim.score(schedule, debug=True)

    starting_demands = {
        2009: (219, 1), 1679: (73, 1), 1427: (160, 1), 1657: (85, 1), 1989: (218, 1), 2843: (63, 1), 1835: (92, 1),
        3613: (158, 1), 2811: (100, 1), 2827: (62, 1), 2951: (107, 1), 3469: (23, 1), 3577: (167, 1), 2845: (39, 1),
        3519: (141, 1), 3545: (38, 1), 1057: (29, 1), 2001: (188, 1), 2979: (148, 1), 2825: (138, 1), 3341: (129, 1),
        1545: (188, 1), 3065: (138, 1), 3335: (109, 1), 2925: (173, 1), 1787: (212, 1), 1963: (201, 1), 3639: (88, 1),
        3647: (78, 1), 7097: (71, 1), 7231: (78, 1), 1719: (148, 1), 2023: (68, 1), 1945: (99, 1), 1775: (229, 1)
    }
    initial_solution = DemandBasedTrafficSignalingSolution({})
    s = MonteCarloBeamSearcher(initial_solution, best)
    s.go(100, 2, 6)

main()