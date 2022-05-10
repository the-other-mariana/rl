import numpy as np
import matplotlib.pyplot as plt

# global variables
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

def print_qsa(vs):
    print("Q(s,a)")
    for a in range(len(qsa)):
        for s in range(len(qsa[0])):
            val = qsa[a][s]
            print("{:.2f}".format(val), end='\t')
        print()

def plot_figs(fig, states, actions, xs, ys, axes):
    y_max = max([max(ys[s]) for s in range(states * actions)])
    y_min = min([min(ys[s]) for s in range(states * actions)])
    for a in range(actions):
        for s in range(states):
            idx = (a * states) + s
            axes[a, s].plot(xs[idx], ys[idx], 'o', linestyle='-')
            axes[a, s].set_title(f"Q(s{s}, a{a})")
            axes[a, s].set_ylim(y_min - (y_min * 0.05), y_max + (y_max * 0.05))
    fig.tight_layout()
    plt.savefig('q-' + 'non'*(not deterministic) + 'det.png', dpi=500)
    plt.show()

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

    qsa = np.zeros((actions, states), dtype=float)
    delta = 1000000 * np.ones((actions, states), dtype=float)
    done = np.zeros((actions, states), dtype=bool)
    iter = 1

    fig = plt.figure( figsize=(18,6) )
    axes = fig.subplots(actions, states)
    xs = [[] for i in range(states * actions)]
    ys = [[] for i in range(states * actions)]


    while(True):

        # stop condition
        for i in range(actions):
            for j in range(states):
                if delta[i, j] < eps:
                    done[i, j] = True
        if all(list(done.reshape(actions * states))):
            break

        # print head and current qsa
        print_head(iter, fr)
        print_qsa(qsa)
        for a in range(actions):
            for s in range(states):
                xs[(a * states) + s].append(iter)
                ys[(a * states) + s].append(qsa[a, s])

        # calculate q(s,a) for each s
        for ai in range(actions):
            for si in range(states):
                sf = fmt[si, ai]
                r = fr[int(sf)]
                terms = []
                # for each q(s,a) cell, we need maximum taking every action a
                for a in range(actions):
                    term = qsa[a, sf]
                    terms.append(term)
                # take the maximum to complete q(s,a) new value
                q = r + gamma * max(terms)
                # difference
                delta[ai][si] = abs(q - qsa[ai][si])
                # update v for next iteration
                qsa[ai][si] = q
        # print new vs
        print_qsa(qsa)
        iter += 1

    print("Optimal Politic:")
    for i in range(states):
        # take max of the column in state s
        action = max(list(qsa[:, i]))
        print(f"s{i} = a{list(qsa[:, i]).index(action)},", end='\t')
    print()
    plot_figs(fig, states, actions, xs, ys, axes)
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
    print(pmt)
    print()
    print("fr:")
    print(fr)

    qsa = np.zeros((actions, states), dtype=float)
    delta = 1000000 * np.ones((actions, states), dtype=float)
    done = np.zeros((actions, states), dtype=bool)
    iter = 1

    fig = plt.figure(figsize=(18, 6))
    axes = fig.subplots(actions, states)
    xs = [[] for i in range(states * actions)]
    ys = [[] for i in range(states * actions)]

    while (True):

        # stop condition
        for i in range(actions):
            for j in range(states):
                if delta[i, j] < eps:
                    done[i, j] = True
        if all(list(done.reshape(actions * states))):
            break

        # print head and current qsa
        print_head(iter, fr)
        print_qsa(qsa)
        for a in range(actions):
            for s in range(states):
                xs[(a * states) + s].append(iter)
                ys[(a * states) + s].append(qsa[a, s])

        # calculate q(s,a) for each s
        for ai in range(actions):
            for si in range(states):
                q = 0.0
                for sf in range(states):
                    p = p = pmt[sf, si, ai]
                    r = fr[int(sf)]

                    terms = []
                    # for each q(s,a) cell, we need maximum taking every action a
                    for a in range(actions):
                        term = qsa[a, sf]
                        terms.append(term)
                    # take the maximum to complete q(s,a) new value
                    q += p * (r + gamma * max(terms))
                # difference
                delta[ai][si] = abs(q - qsa[ai][si])
                # update v for next iteration
                qsa[ai][si] = q
        # print new vs
        print_qsa(qsa)
        iter += 1

    print("Optimal Politic:")
    for i in range(states):
        # take max of the column in state s
        action = max(list(qsa[:, i]))
        print(f"s{i} = a{list(qsa[:, i]).index(action)},", end='\t')
    print()
    plot_figs(fig, states, actions, xs, ys, axes)