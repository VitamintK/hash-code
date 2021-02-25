from collections import namedtuple, deque, defaultdict


Street = namedtuple("Street", ['start', 'end', 'name', 'duration', 'id'])
D, I, S, V, F = map(int, input().split())
streets = []
street_name_to_street_id = dict()
for i in range(S):
    s, e, n, d = input().split()
    streets.append(Street(int(s), int(e), n, int(d), i))
    street_name_to_street_id[n] = i
cars = []
for i in range(V):
    inp = input().split()
    path = inp[1:]
    cars.append(deque([street_name_to_street_id[x] for x in path]))


def intersections_of_path(street_ids):
    return [streets[sid].end for sid in street_ids]


def print_schedule(schedule):
    for inter in schedule:
        print("Intersection", inter.id)
        for street_id, duration in inter.schedule:
            print("Street", streets[street_id].id, ", Duration", duration)
        print("--")


IS = namedtuple("IS", ['id', 'schedule', 'period'])
schedule = [IS(1, [(1, 1), (2, 1)], 2), IS(0, [(0, 1)], 1), IS(2, [(4, 1)], 1)]


def get_light(IS_idx, sec):
    sched = schedule[IS_idx]
    sec = sec % sched.period
    elapsed = 0
    for i, (street_id, time) in enumerate(sched.schedule):
        if elapsed == sec:
            return i
        elapsed += time


def score():
    d = defaultdict(deque)  # map street with id s to idle cars waiting at end of s
    # load cars to their starting interection
    for car in cars:
        start_street_id = car.popleft()
        d[start_street_id].append(car)

    score = 0
    for sec in range(D):
        for i, inter_sched in enumerate(schedule):
            street_id, time = inter_sched.schedule[get_light(i, sec)]
            if not d[street_id]:
                continue

            car = d[street_id].popleft()
            next_street_id = car.popleft()
            if not car:
                score += F + (D - sec)
            else:
                d[next_street_id].append(car)
    return score


print(score())
