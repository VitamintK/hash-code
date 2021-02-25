from collections import namedtuple, defaultdict
Street = namedtuple("Street", ['start', 'end', 'name', 'duration', 'id'])
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
    cars.append(nice_path)

def intersections_of_path(street_ids):
    return [street_ids[0].start] + [streets[sid].end for sid in street_ids]

#intersection-schedule:
IS = namedtuple("IS", ['id', 'schedule'])

for car in cars:
    for street in car[:-1]:
        intersections[streets[street].end].incoming[street] += 1

def print_schedule(schedule):
    print(len(schedule))
    for inter in schedule:
        print(inter.id)
        print(len(inter.schedule))
        for street_id, duration in inter.schedule:
            print(streets[street_id].name, duration)

import math

schedule = []
for i in range(I):
    items = list(intersections[i].incoming.items())
    items.sort(key=lambda sv: -sv[1])
    # items = items[:15]
    denom = sum(v for s,v in items)
    if denom == 0:
        continue
    mini = min(v for s,v in items)
    g = [(street, min(int(volume/mini), 10)) for street,volume in items]
    g = [(s,d) for s,d in g if d > 0]
    schedule.append(IS(i, g))

print_schedule(schedule)