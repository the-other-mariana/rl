import numpy as np

deterministic = True
gamma = 0.9
actions = 2
states = 5
eps = 0.1

def print_head(iter, fr):
    print("==========================")
    print(f"iteration {iter}:")
    print("fr:", fr)
    for i in range(len(list(fr))):
        print(i, end='\t')
    print()

def print_vs(vs):
    for i in range(len(vs)):
        val = vs[i]
        print("{:.2f}".format(val[0]), end='\t')
    print()

if deterministic:

    fmt = [[0, 1],
           [0, 2],
           [1, 3],
           [2, 4],
           [3, 4]]
    fmt = np.array(fmt)
    fr = [-10, 0, -0.04, -0.04, 10]
    print("fmt:")
    print(fmt)
    print("fr:")
    print(fr)

    vs = [(0.0, 0) for s in range(states)]
    delta = [1000000 for s in range(states)]
    done = [False for s in range(states)]
    iter = 1
    while(True):
        # stop condition
        for d in range(states):
            if delta[d] < eps:
                done[d] = True
        if all(done):
            break

        # print head and current vs
        print_head(iter, fr)
        print_vs(vs)

        # calculate v(s) for each s
        for s in range(states):
            terms = []
            r_max = 0
            a_max = 0
            # for each v(s) cell, we need maximum taking every action a
            for a in range(actions):
                sf = fmt[s, a]
                r = fr[int(sf)]
                term = r + gamma * vs[sf][0]
                terms.append(term)
                if term > r_max:
                    a_max = a
                    r_max = term
            # take the maximum, save the action
            v = (max(terms), a_max)
            # difference
            delta[s] = abs(v[0] - vs[s][0])
            # update v for next iteration
            vs[s] = v

        # print new vs
        print_vs(vs)

        iter += 1
    print("Optimal Politic:")
    for i in range(states):
        print(f"s{i} = a{vs[i][1]},", end='\t')
    print()
else:
    pmt = [
           [[0.0, 0.0],
            [0.8, 0.2],
            [0.0, 0.0],
            [0.2, 0.8],
            [0.0, 0.0]],
           [[0.2, 0.8],
            [0.0, 0.0],
            [0.8, 0.2],
            [0.0, 0.0],
            [0.0, 0.0]],
           [[0.0, 0.0],
            [0.2, 0.8],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.8, 0.2]],
           [[0.8, 0.2],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.8, 0.2],
            [0.0, 0.0]],
           [[0.0, 0.0],
            [0.0, 0.0],
            [0.2, 0.8],
            [0.0, 0.0],
            [0.2, 0.8]],
           ]
    pmt = np.array(pmt)
    fr = [-10, 0, -0.04, -0.04, 10]
    print("pmt:")
    for i in range(len(pmt)):
        print(pmt[i],end='\t')
    print()
    print("fr:")
    print(fr)

    vs = [(0.0, 0) for s in range(states)]
    delta = [1000000 for s in range(states)]
    done = [False for s in range(states)]
    iter = 1
    while (True):
        # stop condition
        for d in range(states):
            if delta[d] < eps:
                done[d] = True
        if all(done):
            break

        # print head and current vs
        print_head(iter, fr)
        print_vs(vs)

        # calculate v(s) for each s
        for s in range(states):
            terms = []
            r_max = 0
            a_max = 0
            # for each v(s) cell, we need maximum taking every action a
            for a in range(actions):
                # now each term is the sum of len(states) terms containing p * (r + gamma * v(sf))
                term = 0.0
                for sf in range(states):
                    p = pmt[sf, s, a]
                    r = fr[int(sf)]
                    term += p * (r + gamma * vs[sf][0])
                terms.append(term)
                if term > r_max:
                    a_max = a
                    r_max = term
            v = (max(terms), a_max)
            delta[s] = abs(v[0] - vs[s][0])
            vs[s] = v

        # print new vs
        print_vs(vs)

        iter += 1
    print("Optimal Politic:")
    for i in range(states):
        print(f"s{i} = a{vs[i][1]},", end='\t')
    print()