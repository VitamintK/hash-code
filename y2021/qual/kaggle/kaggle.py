from multiprocessing import Pool
from collections import namedtuple, defaultdict, deque
from enum import Enum
import math
import copy
import random
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# from scipy.optimize import linear_sum_assignment
# import numpy as np

directory = 'data'
PARALLELIZATION = 3

InputData = namedtuple("InputData", ['D', 'I', 'S', 'V', 'F', 'streets', 'cars', 'intersections'])
Street = namedtuple("Street", ['start', 'end', 'name', 'duration', 'id'])
Car = namedtuple("Car", ["id", "path"])
Intersection = namedtuple("Intersection", "incoming")

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
        schedule = Schedule(dict(), self.streets)
        fixed_demands_by_intersection = defaultdict(dict)
        for street_id, (time, duration) in fixed_demands.items():
            fixed_demands_by_intersection[self.streets[street_id].end][street_id] = (time,duration)
        for intersection_id, intersection in enumerate(self.intersections):
            if len(intersection.incoming) == 0:
                continue
            schedule[intersection_id] = IntersectionScheduleGreedy(
                intersection_id,
                [(None, 1) for i in intersection.incoming],
                list(intersection.incoming)
            )
            # to speed this up, this could all be done at build-time in one pass:
            # TODO: NEEDS TO BE SORTED BY PERIODIC TIME, NOT ABSOLUTE TIME
            intersection_fixed_demands = fixed_demands_by_intersection[intersection_id]
            intersection_period = len(intersection.incoming) - len(fixed_demands_by_intersection[intersection_id]) + sum(d for t,d in intersection_fixed_demands.values())
            for street_id, (time, duration) in sorted(intersection_fixed_demands.items(), key=lambda x: x[1][0]%intersection_period):
                if street_id in intersection.incoming:
                    if schedule[intersection_id].time_is_unset(time):
                        schedule[intersection_id].set(street_id, time, duration)
                    # else:
                    #     print(street_id, time, duration, ':', schedule[intersection_id].street_id_at_time(time))
            schedule[intersection_id].make_cache()
        print('building iworld')
        self.iworld = IWorld(self.streets, self.cars, self.intersections)
        score = 0
        print('beginning sim')
        for d in range(self.D):
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
    def is_green(self, street_id, time):
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
def f(args):
    obj, temperature = args
    return obj.sample_neighbor(temperature)

class Solution:
    def sample_neighbors(self, n: int, temperature):
        with Pool(PARALLELIZATION) as p:
            return p.map(f, [(self, temperature) for i in range(n)])
        # return (self.sample_neighbor(temperature) for i in range(n)) 
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
        temperature_function = lambda x: (1 - x/iterations ) * 0.3
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
    def __init__(self, inputs, fixed_demands):
        self.inputs = inputs
        D,I,S,V,F,streets,cars,intersections = self.inputs
        self.sim = Sim([D,I,S,V,F], streets, cars, intersections)
        self.fixed_demands = fixed_demands
        self.schedule = self.sim.get_greedy_schedule(fixed_demands)
    def pretty(self):
        return str(self.fixed_demands)
    def repr(self):
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
        if random.random() < 0.35:
            lb, ub = 0, 2
            k = min(int(get_value(lb,ub)), len(new_fixed_demands))
            for i in random.sample(list(new_fixed_demands),k):
                new_fixed_demands.pop(i)
        # maybe move around some existing demands
        if random.random() < get_value(0.15,0.35):
            lb, ub = 1, 3
            k = min(int(get_value(lb,ub)), len(new_fixed_demands))
            earlier_delta = int(-10 * temperature)
            later_delta = int(40*temperature)
            for i in random.sample(list(new_fixed_demands), k):
                time, duration = new_fixed_demands[i]
                new_fixed_demands[i] = (time+random.randint(earlier_delta,later_delta), duration)
        # maybe change the durations of some existing demands
        if random.random() < get_value(0.15,0.35):
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
            # TODO: DON'T USE TIME%PERIOD HERE -- JUST USE THE RAW TIME
            # %intersection.period
            new_fixed_demands[street_id] = (time, get_random_duration())
        # done
        print(new_fixed_demands)
        return DemandBasedTrafficSignalingSolution(self.inputs, new_fixed_demands)
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
        with open(f'{directory}/kaggle.out', 'w') as f:
            f.write(self.serialize())
        with open(f'{directory}/kaggle_repr.out', 'w') as f:
            f.write(self.repr())
        with open(f'{directory}/kaggle_score.out', 'w') as f:
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
        with open(f'{directory}/kaggle.out', 'w') as f:
            f.write(self.serialize())
        with open(f'{directory}/kaggle_score.out', 'w') as f:
            f.write(str(self.score()))

def main(inputs):
    try:
        with open(f'{directory}/kaggle_score.out', 'r') as f:
            best = int(f.readline())
    except FileNotFoundError:
        with open(f'{directory}/kaggle_score.out', 'w') as f:
            f.write('0')
        best = 0

    try:
        with open(f'{directory}/kaggle_repr.out', 'r') as f:
            starting_demands = eval(f.readline())
    except FileNotFoundError:
        starting_demands = dict()

    D, I, S, V, F, streets, cars, intersections = inputs
    # sim = Sim([D,I,S,V,F], streets, cars, intersections)
    # schedule = sim.get_greedy_schedule(dict())
    # sim.score(schedule, debug=True)
    # starting = {27390: (8, 2), 34269: (28, 1), 39289: (10, 1), 39297: (9, 1), 15194: (9, 3), 1241: (36, 2), 8373: (29, 3), 8455: (22, 1), 8449: (34, 2), 8962: (3, 2), 28859: (16, 2), 28829: (5, 1), 10014: (29, 1), 29692: (0, 1), 40945: (11, 2), 19001: (17, 1), 18981: (3, 2), 18993: (5, 3), 45864: (11, 2), 50897: (12, 3), 50889: (1, 1), 39884: (7, 1), 46613: (20, 2), 46611: (18, 1), 6367: (70, 1), 6293: (45, 1), 6373: (41, 1), 22069: (0, 1), 22075: (15, 1), 17917: (7, 2), 17931: (13, 1), 17953: (8, 2), 33092: (3, 2), 18490: (1, 2), 58995: (4, 2), 8475: (10, 1), 8471: (19, 1), 49110: (4, 1), 59362: (0, 2), 48807: (3, 1), 48817: (11, 2), 26307: (10, 3), 6756: (2, 2), 10469: (56, 1), 10571: (56, 2), 3999: (81, 1), 4095: (62, 1), 17926: (2, 1), 3722: (0, 1), 16603: (2, 2), 33043: (11, 1), 26449: (20, 1), 6481: (33, 1), 30516: (9, 2), 1835: (230, 1), 29861: (25, 1), 24963: (8, 1), 44336: (6, 2), 23671: (18, 1), 30967: (10, 1), 49742: (4, 1), 48423: (3, 2), 28082: (2, 3), 10205: (6, 1), 9987: (20, 1), 2527: (26, 1), 20812: (0, 1), 15281: (0, 1), 52691: (1, 1), 2991: (139, 1), 7215: (32, 1), 34111: (3, 1), 9251: (13, 1), 4980: (8, 1), 4325: (34, 1), 44329: (3, 1)}
    # starting_demands = {4779: (5, 1), 4755: (-2, 2), 4767: (38, 2), 2353: (2, 1), 2351: (25, 2), 3063: (113, 1), 13815: (14, 2), 2854: (3, 1), 50121: (14, 2), 15763: (44, 1), 25229: (10, 1), 49537: (2, 1), 2473: (48, 1), 12749: (28, 1), 30817: (13, 1), 30835: (11, 2), 13817: (35, 1), 13831: (19, 1), 29695: (9, 2), 57575: (2, 1), 53656: (-3, 2), 3201: (24, 1), 3217: (51, 2), 60293: (5, 2), 14801: (7, 1), 4620: (6, 1), 39183: (5, 1), 59074: (3, 2), 3282: (12, 1), 22617: (4, 2), 44802: (1, 1), 38374: (3, 1), 21600: (3, 2), 35791: (6, 1), 4882: (2, 1), 30730: (2, 2), 12353: (43, 1), 12858: (2, 1), 5311: (88, 1), 19753: (1, 2), 17451: (6, 1), 11997: (18, 1), 36267: (10, 1), 47731: (6, 1), 4677: (78, 1), 48815: (1, 1), 58624: (3, 2), 38727: (3, 2), 25177: (13, 1), 15619: (48, 1), 55296: (2, 1), 28728: (5, 1), 5804: (0, 1), 18982: (3, 2), 38012: (2, 1), 44861: (1, 2), 20822: (1, 2), 17246: (0, 2), 60363: (12, 2), 48329: (1, 2), 28080: (2, 1), 19986: (5, 1), 5130: (1, 2), 27953: (4, 2), 7827: (27, 1), 24251: (2, 2), 35833: (7, 1), 55741: (0, 2), 3971: (59, 1), 31843: (13, 1), 15907: (13, 1), 45854: (7, 1), 59982: (1, 1), 18760: (2, 2), 9257: (24, 1), 829: (55, 1), 2791: (182, 1), 29423: (0, 1), 1796: (1, 2)}
    # starting_demands = {4779: (5, 1), 4755: (-2, 2), 4767: (38, 2), 2353: (2, 1), 2351: (25, 2), 3063: (113, 1), 13815: (14, 2), 2854: (3, 1), 50121: (14, 2), 15763: (44, 1), 25229: (10, 1), 49537: (9, 1), 2473: (48, 1), 12749: (28, 1), 30817: (13, 1), 30835: (11, 2), 13817: (35, 1), 13831: (19, 1), 29695: (16, 2), 57575: (9, 1), 53656: (-3, 2), 3201: (24, 1), 3217: (51, 2), 60293: (5, 2), 14801: (7, 1), 4620: (6, 1), 39183: (5, 1), 59074: (3, 2), 3282: (12, 1), 22617: (4, 2), 44802: (1, 1), 38374: (3, 1), 21600: (3, 2), 35791: (6, 1), 4882: (2, 1), 30730: (2, 1), 12353: (43, 1), 12858: (7, 1), 5311: (88, 1), 19753: (1, 2), 17451: (6, 1), 11997: (18, 1), 36267: (10, 1), 47731: (6, 1), 4677: (78, 1), 48815: (1, 1), 58624: (3, 2), 38727: (3, 2), 25177: (13, 1), 15619: (48, 1), 55296: (2, 1), 28728: (5, 1), 5804: (0, 1), 18982: (3, 2), 38012: (2, 1), 44861: (1, 2), 20822: (1, 2), 17246: (0, 2), 60363: (12, 2), 48329: (1, 2), 28080: (9, 1), 19986: (5, 1), 5130: (1, 2), 27953: (4, 2), 7827: (27, 1), 24251: (2, 2), 35833: (7, 1), 55741: (0, 2), 3971: (59, 1), 31843: (13, 1), 15907: (13, 1), 45854: (7, 1), 59982: (1, 1), 18760: (2, 2), 9257: (24, 1), 829: (63, 1), 2791: (182, 1), 29423: (0, 1), 1796: (1, 2), 9026: (2952, 2), 10701: (3411, 1), 25649: (3380, 1), 38744: (3034, 1), 24503: (2943, 1), 157: (3359, 1), 50733: (1702, 1), 5734: (1268, 2), 54145: (670, 1), 15059: (2328, 1), 19207: (2054, 1), 34330: (1456, 2), 21607: (3710, 1), 46415: (3475, 2), 4617: (1461, 3), 17495: (3057, 1), 31746: (1755, 1), 42737: (2733, 1), 45407: (3490, 2), 57946: (2040, 1), 15103: (2488, 1), 5727: (3360, 1), 18750: (1653, 2), 3769: (2808, 1), 51597: (2367, 1), 22245: (3310, 1), 47851: (2914, 1), 24029: (2019, 1), 14144: (2385, 1), 24637: (435, 1), 47682: (1725, 1), 5637: (2430, 1), 23723: (3558, 1)}
    # starting_demands = {4779: (5, 1), 4755: (-2, 2), 4767: (38, 2), 2353: (2, 1), 2351: (25, 2), 3063: (113, 1), 13815: (14, 2), 2854: (3, 1), 50121: (14, 2), 15763: (44, 1), 25229: (10, 1), 49537: (9, 1), 2473: (48, 1), 12749: (28, 1), 30817: (13, 1), 30835: (11, 2), 13817: (35, 1), 13831: (27, 1), 29695: (16, 2), 57575: (9, 1), 53656: (-3, 2), 3201: (24, 1), 3217: (51, 2), 60293: (5, 2), 14801: (7, 1), 4620: (6, 1), 39183: (6, 1), 59074: (3, 2), 3282: (12, 1), 22617: (3382, 1), 44802: (1, 1), 38374: (3, 1), 21600: (3, 2), 35791: (6, 1), 4882: (2, 1), 30730: (9, 1), 12353: (43, 1), 12858: (7, 1), 5311: (88, 1), 19753: (1, 2), 17451: (11, 1), 11997: (18, 1), 36267: (14, 1), 47731: (6, 1), 4677: (78, 1), 48815: (4, 1), 58624: (3, 2), 38727: (3, 2), 25177: (13, 1), 15619: (49, 1), 55296: (2, 1), 28728: (5, 1), 5804: (0, 1), 18982: (3, 2), 38012: (2, 1), 44861: (1, 2), 20822: (2, 2), 17246: (0, 2), 60363: (12, 2), 48329: (1, 2), 28080: (9, 1), 19986: (5, 1), 5130: (1, 1), 27953: (4, 2), 7827: (27, 1), 24251: (2, 2), 35833: (7, 1), 55741: (0, 2), 3971: (59, 1), 31843: (13, 1), 15907: (12, 1), 45854: (7, 1), 59982: (1, 1), 18760: (2, 2), 9257: (24, 1), 829: (67, 1), 2791: (182, 1), 29423: (0, 1), 1796: (1, 2), 9026: (2956, 2), 10701: (3411, 1), 25649: (3380, 1), 38744: (3035, 1), 24503: (2943, 1), 157: (3359, 1), 50733: (1702, 1), 5734: (1268, 2), 54145: (670, 1), 15059: (2328, 1), 19207: (2063, 1), 34330: (1464, 2), 21607: (3710, 1), 46415: (3475, 2), 4617: (1461, 3), 17495: (3057, 1), 31746: (1755, 1), 42737: (2733, 1), 45407: (3490, 2), 57946: (2039, 1), 15103: (2488, 1), 5727: (3360, 1), 18750: (1653, 1), 3769: (2808, 1), 51597: (2367, 1), 22245: (3310, 1), 47851: (2914, 1), 24029: (2019, 1), 14144: (2385, 1), 24637: (435, 1), 47682: (1732, 1), 5637: (2430, 1), 23723: (3558, 1), 19204: (3313, 1), 20367: (3018, 1), 54299: (2554, 2), 7125: (600, 1), 53783: (3836, 1), 42062: (2916, 1), 19133: (3136, 1), 57764: (2136, 3), 4005: (2934, 1), 59157: (2781, 2), 12789: (3482, 1), 18661: (3214, 1), 7373: (2710, 1), 3627: (1445, 1), 47603: (2857, 1), 20833: (1705, 1), 13440: (3090, 1), 16901: (1738, 1), 25195: (3157, 1), 14771: (3408, 1), 6455: (3399, 1), 24396: (2358, 1), 31759: (3117, 1), 41222: (3083, 1), 62851: (1426, 2), 6533: (3715, 1), 15774: (2247, 1), 1109: (3325, 1), 56470: (2905, 2), 19831: (3655, 1), 24076: (1389, 2), 23464: (3070, 1), 8335: (3627, 1), 33151: (3565, 1), 14531: (3069, 1), 15858: (2263, 1), 41975: (2867, 1), 27918: (2651, 2), 123: (2522, 1), 9471: (3204, 1), 44549: (2094, 1), 35347: (3338, 1), 26309: (3489, 1), 23615: (1518, 2), 57499: (3379, 2), 3839: (1440, 1), 41345: (2900, 1), 30406: (1076, 2), 47823: (3193, 1), 38632: (1795, 2), 34315: (385, 1), 20202: (2258, 1), 32785: (872, 2), 62059: (3320, 2), 20427: (2914, 1), 15191: (3272, 1), 50656: (1323, 1), 10805: (3303, 1), 37020: (2956, 3), 45829: (994, 1), 33789: (2566, 1), 29107: (2951, 1), 18525: (3239, 1), 31795: (1334, 1), 10321: (3518, 1), 473: (1344, 1), 3663: (2713, 1), 3883: (4082, 1), 16717: (3769, 1), 53941: (2907, 1), 33321: (2358, 1), 13427: (3226, 1), 25130: (2419, 2), 49585: (2724, 1), 13099: (2165, 1), 45101: (1730, 1), 1647: (1134, 1), 28417: (1653, 1), 1372: (973, 1), 53596: (3180, 1), 10423: (2995, 1), 10047: (3473, 1), 4435: (3221, 1), 13616: (1251, 1), 48321: (2608, 2), 37458: (3384, 2), 15492: (1432, 1), 24552: (1330, 1), 5377: (2548, 1), 2779: (3485, 1), 27751: (3397, 2), 14667: (2745, 1), 1675: (3315, 1), 27678: (1847, 1), 32395: (3064, 1), 60447: (1722, 1), 8577: (3768, 1), 39880: (1409, 1), 20447: (2704, 1), 19817: (1984, 1), 54445: (2452, 2), 59780: (2222, 1), 29251: (2428, 1), 12496: (2775, 1), 7209: (2069, 1), 10784: (2589, 1), 828: (3053, 1), 18715: (3397, 1), 4135: (3046, 1), 2979: (3762, 1), 30523: (2798, 1), 32329: (2761, 1), 8261: (3513, 1), 20812: (1385, 1), 3043: (2148, 1), 2553: (2788, 1), 34648: (3110, 1), 39942: (994, 2), 18841: (2689, 1), 4643: (2581, 2), 58429: (1528, 1), 44077: (2942, 1), 49013: (3291, 1), 53382: (3404, 1), 55501: (2076, 1), 21524: (2821, 1), 56511: (1570, 2), 10415: (3219, 1), 19776: (1106, 1), 57023: (3821, 1), 5583: (2429, 1), 33046: (2715, 2), 13670: (2380, 1), 62501: (821, 1), 62949: (2595, 2), 57361: (983, 2), 891: (2241, 1), 43925: (3567, 1), 7293: (3288, 1), 18393: (3526, 1), 33005: (3091, 2), 29853: (3041, 1), 33836: (1822, 1), 21459: (3537, 1), 28235: (3207, 1), 21144: (3672, 1), 19205: (2373, 1), 16949: (2757, 1), 8815: (2679, 1), 13555: (2687, 1), 36023: (2628, 1), 2855: (2501, 1), 61428: (821, 1), 8625: (1944, 1), 2951: (2654, 1), 26007: (2972, 1), 14019: (3004, 1), 18637: (3658, 1), 5413: (2643, 1), 6917: (3253, 1), 8202: (2401, 3), 12978: (1924, 1), 31399: (3233, 1), 12712: (2901, 1), 3081: (2697, 1), 8875: (1322, 1), 37101: (2988, 1), 49883: (3237, 3), 61563: (3403, 1), 26100: (753, 1), 61725: (3009, 1), 15144: (2990, 1), 41207: (3747, 1), 5961: (2024, 1), 18127: (2536, 1), 39060: (2794, 2), 32209: (1976, 2), 17109: (3452, 1), 41381: (1562, 1), 38018: (1811, 1), 46968: (956, 1), 17699: (3580, 1), 62569: (1678, 1), 4195: (1947, 1), 48312: (777, 1), 8413: (3480, 1), 49375: (2911, 1), 55105: (1014, 2), 561: (3201, 1), 56837: (700, 2), 11656: (1198, 1), 3129: (2049, 1), 31033: (2361, 2), 34541: (1149, 1), 8467: (2266, 1), 55275: (1440, 1), 51173: (2161, 1), 3751: (3174, 1), 55283: (2954, 3), 62533: (2065, 2), 406: (3139, 2), 534: (3951, 1), 11285: (3634, 1), 3367: (1097, 1), 10491: (1241, 1), 4455: (2903, 1), 13596: (2623, 1)}
    initial_solution = DemandBasedTrafficSignalingSolution(inputs, starting_demands)
    s = MonteCarloBeamSearcher(initial_solution, best)
    s.go(900, 2, 6)

if __name__ == '__main__':
    #### read in input ###########
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
    inputs = InputData(D, I, S, V, F, streets, cars, intersections)
    main(inputs)