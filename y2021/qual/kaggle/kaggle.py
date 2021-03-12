from collections import namedtuple, defaultdict, deque
from enum import Enum
import math
import copy
import random
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

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
    
    def get_greedy_schedule(self):
        schedule = Schedule(dict(), streets)
        for intersection_id, intersection in enumerate(self.intersections):
            schedule[intersection_id] = IntersectionScheduleGreedy(
                intersection_id,
                [(None, 1) for i in intersection.incoming],
                list(intersection.incoming)
            )
        print('building iworld')
        self.iworld = IWorld(streets, cars, intersections)
        score = 0
        print('beginning sim')
        for d in range(self.D):
            updates = []
            for icar in self.iworld.icars:
                if icar.status == CarStatus.WAITING:
                    street_id = icar.car.path[icar.street_of_path]
                    queue = self.iworld.queues[street_id]
                    if icar == queue[0]:
                        if schedule.intersection_of(street_id).is_green(street_id, d): 
                            updates.append(Update(icar.car.id))
                        elif schedule.intersection_of(street_id).time_is_unset(d) and schedule.intersection_of(street_id).street_is_unset(street_id):
                            schedule.intersection_of(street_id).set(street_id, d)
                            updates.append(Update(icar.car.id))
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
        for intersection_id in schedule:
            schedule[intersection_id].fill_in_rest()
        return schedule

    def score(self, schedule, debug=False):
        def debugger(*args, **kwargs):
            if debug:
                print(*args, **kwargs)
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
                        if len(queue) > 1:
                            street = self.streets[street_id]
                            debugger('intersection {}: street {} ({}). At time {}, backed up w {} cars'.format(
                                street.end, street_id, street.name, d, len(queue)))
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
        print(n, 'cars,', score, 'score')
        self.score_cache = score
        return score

class IntersectionSchedule:
    def __init__(self, id, schedule: [('street_id', 'duration')]):
        self.id = id
        self.schedule = schedule
    def period(self):
        return sum(duration for street_id, duration in self.schedule)
    def street_id_at_time(self, time):
        time = time%self.period()
        t = 0
        for s_id, duration in self.schedule:
            t += duration
            if t > time:
                return s_id
    def is_green(self, street_id, time):
        # THIS IS REALLY SLOW.  CAN BE AMORTIZED O(1)-ish (persisting a pointer), OR O(LOGN) (binary search each time) INSTEAD OF O(N) where N is number of streets on this intersection
        # or amortized O(1) with O(period) space
        return self.street_id_at_time(time) == street_id
    def mutate(self):
        random.shuffle(self.schedule)

class IntersectionScheduleGreedy(IntersectionSchedule):
    def __init__(self, id, schedule: [('street_id', 'duration')], incoming_streets):
        if schedule is None:
            self.schedule = [(street_id, 1) for street_id in streets]
        self.schedule = schedule
        self.id = id
        self.unused_streets = set(incoming_streets)
    def time_is_unset(self, time):
        return self.street_id_at_time(time) is None
    def street_is_unset(self, street_id):
        return street_id in self.unused_streets
    def set(self, street_id, time):
        assert self.street_id_at_time(time) is None
        self.schedule[time%self.period()] = (street_id, 1)
        self.unused_streets.remove(street_id)
    def fill_in_rest(self):
        iterator = iter(self.unused_streets)
        for i in range(len(self.schedule)):
            if self.schedule[i][0] is None:
                self.schedule[i] = (next(iterator), 1)

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
            if it%1000 == 0:
                for cand in candidates:
                    print(cand.score())
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

class TrafficSignalingSolution(Solution):
    def __init__(self, schedule):
        self.schedule = schedule
        self.sim = Sim([D,I,S,V,F], streets, cars, intersections)
    def serialize(self):
        return self.schedule.serialize()
    def sample_neighbor(self, temperature):
        new_schedule = self.schedule.clone()
        k = int(4 * temperature)
        for intersection_id in random.sample(list(new_schedule), k):
            new_schedule[intersection_id].mutate()
        return TrafficSignalingSolution(new_schedule)
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
    # schedule = Schedule(dict(), streets)
    # for i in range(I):
    #     items = list(intersections[i].incoming.items())
    #     items.sort(key=lambda sv: -sv[1])
    #     # items = items[:15]
    #     denom = sum(v for s,v in items)
    #     if denom == 0:
    #         continue
    #     mini = min(v for s,v in items)
    #     g = [(street, 1) for street,volume in items if volume>0]
    #     # g = [(s,d) for s,d in g if d > 0]
    #     schedule[i] = IntersectionSchedule(i, g)
    sim = Sim([D,I,S,V,F], streets, cars, intersections)
    schedule = sim.get_greedy_schedule()
    sim.score(schedule, debug=True)
    initial_solution = TrafficSignalingSolution(schedule)
    s = MonteCarloBeamSearcher(initial_solution, best)
    s.go(100, 2, 5)

main()