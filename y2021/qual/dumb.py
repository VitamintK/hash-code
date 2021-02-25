from collections import namedtuple
Street = namedtuple("Street", ['start', 'end', 'name', 'duration', 'id'])
Intersection = namedtuple("Intersection", "incoming")
D, I, S, V, F = map(int, input().split())
streets = []
street_name_to_street_id = dict()
intersections = [Intersection(incoming=set()) for i in range(I)]
for i in range(S):
    s, e, n, d = input().split()
    streets.append(Street(int(s), int(e), n, int(d), i))
    intersections[int(e)].incoming.add(i)
    street_name_to_street_id[n] = i
cars = []
for i in range(V):
    inp = input().split()
    path = inp[1:]
    cars.append([street_name_to_street_id[x] for x in path])


def intersections_of_path(street_ids):
    return [streets[sid].end for sid in street_ids]

#intersection-schedule:
IS = namedtuple("IS", ['id', 'schedule'])
schedule = [IS(1, [(2,1), (1, 3)]), IS(0, [(0, 1)]), IS(2, [(4, 1)])]

def print_schedule(schedule):
    print(len(schedule))
    for inter in schedule:
        print(inter.id)
        print(len(inter.schedule))
        for street_id, duration in inter.schedule:
            print(streets[street_id].name, duration)

schedule = []
for i in range(I):
    schedule.append(IS(i, [(x, 1) for x in intersections[i].incoming]))

print_schedule(schedule)