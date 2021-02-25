from collections import namedtuple
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
    cars.append([street_name_to_street_id[x] for x in path])

def intersections_of_path(street_ids):
    return [streets[sid].end for sid in street_ids]
