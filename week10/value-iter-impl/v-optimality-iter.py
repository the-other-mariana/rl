import numpy as np
import matplotlib.pyplot as plt

# global variables
deterministic = False
gamma = 0.6
actions = 2
states = 3
eps = 0.01

def print_head(iter, fr, names_s):
    print("==========================")
    print(f"Iteration {iter}:")
    print("fr:", fr)
    for i in range(len(list(fr))):
        print(names_s[i], end='\t')
    print()

def print_vs(vs, msg):
    print("V(s)", msg)
    for i in range(len(vs)):
        val = vs[i]
        print("{:.2f}".format(val[0]), end='\t')
    print()

def plot_figs(fig, states, xs, ys, axes, names_s):
    y_max = max([max(ys[s]) for s in range(states)])
    y_min = min([min(ys[s]) for s in range(states)])
    for s in range(states):
        axes[s].plot(xs[s], ys[s], 'o', linestyle='-')
        axes[s].set_title(f"V({names_s[s]})")
        axes[s].set_ylim(y_min - (y_min * 0.05), y_max + (y_max*0.05))
    fig.tight_layout()
    plt.savefig('v-hw2-' + 'non'*(not deterministic) + 'det.png', dpi=500)
    plt.show()

if deterministic:

    fmt = [[1, 1],
           [0, 2],
           [2, 0],
           [0, 3]]
    fmt = np.array(fmt)
    fr = [2, 1, -1, 10]
    names_s = ['s1', 's2', 's3', 's4']
    names_a = ['a1', 'a2']
    print("fmt:")
    print(fmt)
    print("fr:")
    print(fr)

    vs = [(0.0, 0) for s in range(states)]
    delta = [1000000 for s in range(states)]
    done = [False for s in range(states)]
    iter = 1

    fig = plt.figure( figsize=(18,3) )
    axes = fig.subplots(1, states)
    xs = [[] for i in range(states)]
    ys = [[] for i in range(states)]


    while(True):

        # stop condition
        for d in range(states):
            if delta[d] < eps:
                done[d] = True
        if all(done):
            break

        # print head and current vs
        print_head(iter, fr, names_s)
        print_vs(vs, 'current')
        for s in range(states):
            xs[s].append(iter)
            ys[s].append(vs[s][0])

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
        print_vs(vs, 'new')

        iter += 1
    print("Optimal Politic:")
    for i in range(states):
        print(f"{names_s[i]} = {names_a[vs[i][1]]},", end='\t')
    print()
    plot_figs(fig, states, xs, ys, axes, names_s)
else:
    pmt = [
           [[0.4, 0.2],
            [0.5, 0.0],
            [1.0, 0.3]],
           [[0.5, 0.8],
            [0.0, 0.0],
            [0.0, 0.6]],
           [[0.1, 0.0],
            [0.5, 1.0],
            [0.0, 0.1]]
           ]
    pmt = np.array(pmt)
    fr = [2, 1, -1]
    names_s = ['s1', 's2', 's3']
    names_a = ['a1', 'a2']
    print("pmt:")
    print(pmt)
    print()
    print("fr:")
    print(fr)

    vs = [(0.0, 0) for s in range(states)]
    delta = [1000000 for s in range(states)]
    done = [False for s in range(states)]
    iter = 1

    fig = plt.figure(figsize=(18, 3))
    axes = fig.subplots(1, states)
    xs = [[] for i in range(states)]
    ys = [[] for i in range(states)]
    while (True):
        # stop condition
        for d in range(states):
            if delta[d] < eps:
                done[d] = True
        if all(done):
            break

        # print head and current vs
        print_head(iter, fr, names_s)
        print_vs(vs, 'current')
        for s in range(states):
            xs[s].append(iter)
            ys[s].append(vs[s][0])

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
        print_vs(vs, 'new')

        iter += 1
    print("Optimal Politic:")
    for i in range(states):
        print(f"{names_s[i]} = {names_a[vs[i][1]]},", end='\t')
    print()
    plot_figs(fig, states, xs, ys, axes, names_s)