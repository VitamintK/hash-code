from collections import namedtuple
import math

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

#for car in cars:
#    print(intersections_of_path(car))



G_inc = {}
G_out = {}
for i in range(I):
    G_inc[i] = {}
    G_out[i] = {}

for street in streets:
    G_inc[street.end][street.start] = [street.id, 0, 0, 0]
    #G_out[street.start][street.end] = [street.name, 0,0]
    
    #.append([street.start, street.name, 0, 0])
    #G_out[street.start].append([street.end, street.name, 0, 0])




for car in cars:
    p = intersections_of_path(car)
    for i in range(0, len(p)-1):
        #pass
        G_inc[p[i+1]][p[i]][1] += 1

#print(G_inc)


for intersection in G_inc:
    s = 0
    m = 10000000
    for node in G_inc[intersection]:
        s += G_inc[intersection][node][1]

    for node in G_inc[intersection]:
        if s == 0:
            break
        G_inc[intersection][node][2] = G_inc[intersection][node][1]/s
        if G_inc[intersection][node][2] > 0 and G_inc[intersection][node][2] < m:
            m = G_inc[intersection][node][2]

    for node in G_inc[intersection]:
        G_inc[intersection][node][3] = math.ceil(G_inc[intersection][node][2]/m)
        


IS = namedtuple("IS", ['id', 'schedule'])

schedule = []
for intersection in G_inc:
    times = []
    for node in G_inc[intersection]:
        if G_inc[intersection][node][3] > 0:
            times.append((G_inc[intersection][node][0], G_inc[intersection][node][3]))

    if len(times) != 0:
        schedule.append(IS(intersection, times))


def print_schedule(schedule):
    print(len(schedule))
    for inter in schedule:
        print(inter.id)
        print(len(inter.schedule))
        for street_id, duration in inter.schedule:
            print(streets[street_id].name, duration)

print_schedule(schedule)