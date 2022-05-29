import numpy as np
import random

deterministic = True
gamma = 0.8
alpha = 0.9

actions = 2
states = 5

eps = 0.1
success_ratio = 0.85
tests = 20
max_actions_per_test = 100

def print_qsa(qsa, msg):
    print("Q(s,a)", msg)
    for a in range(len(qsa)):
        for s in range(len(qsa[0])):
            val = qsa[a][s]
            print("{:.2f}".format(val), end='\t')
        print()

def print_head(iter, fr, names_s, cycle):
    print("==========================")
    print(f"Iteration {iter}, cycle {cycle}:")
    print("fr:", fr)
    for i in range(len(list(fr))):
        print(names_s[i], end='\t')
    print()

def test_agent_det(tests, politic, fmt, goal):
    success = 0

    for t in range(tests):
        s = 1
        at = 0
        while at < max_actions_per_test:
            if s == goal:
                success += 1
                break
            a = politic[s]
            s = fmt[s, a]
            at += 1
    return (success * 1.0) / tests

def test_agent_nondet(tests, politic, pmt, goal):
    success = 0

    for t in range(tests):
        s = 1
        at = 0
        while at < max_actions_per_test:
            if s == goal:
                success += 1
                break
            a = politic[s]
            s = rand_state(pmt, s, a)
            at += 1
    return (success * 1.0) / tests

def rand_state(pmt, s, action):
    ps = pmt[:, s, action]
    idx = np.nonzero(ps) # 0 2
    vals = np.array([ps[i] for i in idx[0]]) # 80 20
    lims = sorted(list(vals * 100)) # 20 80
    #print(lims)
    coin = random.randint(0, 100)
    choice = 0
    for l in range(len(lims)):
        if coin > 0 and coin <= lims[l]:
            choice = list(ps * 100).index(lims[l])
    return choice

def get_r(fr, sf, s, a):
    r = 0
    if len(np.array(fr).shape) == 1:
        # simple fr
        r = fr[int(sf)]
    elif len(np.array(fr).shape) > 1:
        # complex fr
        r = fr[int(sf)][s][a]
    return r

if deterministic:
    fmt = [[1, 0],
           [2, 0],
           [3, 1],
           [4, 2],
           [4, 3]]
    fmt = np.array(fmt)
    fr = [-10, 0, -0.4, -0.4, 10]
    names_s = ['sf1', 's1', 's2', 's3', 'sf2']
    names_a = ['->', '<-']

    init = 1
    final = [0, 4]

    qsa = np.zeros((actions, states), dtype=float)
    for i in range(actions):
        for j in range(states):
            qsa[i, j] = random.uniform(0.05, 0.95)
    #qsa[0, 0] = 0.67
    #qsa[0, 1] = 0.54
    #qsa[1, 1] = 0.15
    politic = [1 for s in range(states)]

    s = 1
    iter = 1
    c = 0
    while True:
        col = qsa[:, s]
        action = list(col).index(max(col))
        sf = fmt[s, action]
        r = get_r(fr, sf, s, action)
        q_max = 0.0
        for a in range(actions):
            if qsa[a, sf] > q_max:
                q_max = qsa[a, sf]
        qsf = r + gamma * q_max
        new_qsa = qsa[action, s] + alpha * (qsf - qsa[action, s])
        print(f'Q({s}, {action}) = {qsa[action, s]} -> {new_qsa}')
        qsa[action, s] = new_qsa
        politic[s] = action
        if s in final:
            # new cycle
            print("[CYCLE] Optimal politic so far:")
            for i in range(states):
                print(f"{names_s[i]} = {names_a[politic[i]]},", end='\t')
            print()
            # send the agent to test the politic
            succ = test_agent_det(tests, politic, fmt, 4)
            if succ >= success_ratio:
                break
            # back to initial state
            s = 1
            c += 1
        else:
            s = sf
        iter += 1
    print_head(iter, fr, names_s, c)
    print_qsa(qsa, 'final')
    print("[FINAL] Optimal politic so far:")
    for si in range(states):
        print(f"{names_s[si]} = {names_a[politic[si]]},", end='\t')
    print()
else:
    pmt = [
        [[0.2, 0.8],
         [0.2, 0.8],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0]],
        [[0.8, 0.2],
         [0.0, 0.0],
         [0.2, 0.8],
         [0.0, 0.0],
         [0.0, 0.0]],
        [[0.0, 0.0],
         [0.8, 0.2],
         [0.0, 0.0],
         [0.2, 0.8],
         [0.0, 0.0]],
        [[0.0, 0.0],
         [0.0, 0.0],
         [0.8, 0.2],
         [0.0, 0.0],
         [0.2, 0.8]],
        [[0.0, 0.0],
         [0.0, 0.0],
         [0.0, 0.0],
         [0.8, 0.2],
         [0.8, 0.2]]
    ]
    pmt = np.array(pmt)
    fr = [-10, 0, -0.4, -0.4, 10]
    names_s = ['sf1', 's1', 's2', 's3', 'sf2']
    names_a = ['->', '<-']

    init = 1
    final = [0, 4]

    qsa = np.zeros((actions, states), dtype=float)
    for i in range(actions):
        for j in range(states):
            qsa[i, j] = random.uniform(0.05, 0.95)
    politic = [1 for s in range(states)]

    s = 1
    iter = 1
    c = 0
    while True:
        col = qsa[:, s]
        action = list(col).index(max(col))
        sf = rand_state(pmt, s, action)
        # q_max is the max(af)(sf, af) term
        q_max = 0.0
        for a in range(actions):
            if qsa[a, sf] > q_max:
                q_max = qsa[a, sf]
        # now qsf is the sum of s terms of p * (r + gamma * q_max)
        qsf = 0.0
        for sfi in range(states):
            p = pmt[sfi, s, action]
            r = get_r(fr, sfi, s, action)
            qsf += p * (r + gamma * q_max)
        new_qsa = qsa[action, s] + alpha * (qsf - qsa[action, s])
        print(f'Q({s}, {action}) = {qsa[action, s]} -> {new_qsa}')
        qsa[action, s] = new_qsa
        politic[s] = action

        if s in final:
            # new cycle
            print("[CYCLE] Optimal politic so far:")
            for i in range(states):
                print(f"{names_s[i]} = {names_a[politic[i]]},", end='\t')
            print()
            # send the agent to test the politic
            succ = test_agent_nondet(tests, politic, pmt, 4)
            if succ >= success_ratio:
                break
            # back to initial state
            s = 1
            c += 1
        else:
            s = sf
        iter += 1
    print_head(iter, fr, names_s, c)
    print_qsa(qsa, 'final')
    print("[FINAL] Optimal politic so far:")
    for si in range(states):
        print(f"{names_s[si]} = {names_a[politic[si]]},", end='\t')
    print()