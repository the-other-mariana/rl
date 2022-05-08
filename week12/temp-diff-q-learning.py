# RL: Q-Learning [Q] using Temporary Difference DT
#Q(s,a): Table
#PROBLEMA: Frozen Lake
#API: -

import gym
import numpy as np

# states =0,1,...,15
# actions =0,1,2,3
α = 0.2        # learning ration
γ = 0.9        # discount factor

#=======================================================
# same Q class created before
class Q:
    def __init__(self, Ns, Na):
        self.q  = np.zeros((Ns, Na))
        self.Ns = Ns
        self.Na = Na

    def update(self, s, a, q):
        self.q[s][a] = q

    def get(self, s, a):
        return self.q[s][a]

    # get max based on a state
    def maximum(self, s):
        return max(self.q[s])

    # get the action to execute based on which a has max Q
    def action(self, s):
        a_max = 0
        qs_max = -float('inf')
        for a in range(self.Na):
            if (self.q[s][a] > qs_max):
                a_max = a
                qs_max = self.q[s][a]
        return a_max

    # get the Q matrix (2D)
    def getQ(self):
        return self.q

#=======================================================
class Agent:
    def __init__(self, Ns, Na):
        self.Ns = Ns             #No. of states
        self.Na = Na             #No. of actions
        self.q  = Q(Ns, Na)      #Q(s,a)

    # this function is called per transition, since it is update per transition
    def learn(self, s, env):
        # LEARNING: random action, instead of choosing which Q is bigger we get a random action choice?
        a = env.action_space.sample()
        sf, r, is_done, _ = env.step(a)

        # update equations
        # LEARNING: Bellman's Optimality Equations
        # Q(s,a) = self.q.get(s,a)
        qsa = self.q.get(s,a) + α*((r + γ*self.q.maximum(sf)) - self.q.get(s,a))
        self.q.update(s,a,qsa)

        if (is_done):
            sf = env.reset()
        return sf

    def action(self, s):
        return self.q.action(s)

    # this executes one episode: to test the agent in one episode
    # if we want to test 20 times, we call this 20 times
    def run(self, env):
        p = 0
        r_total = 0.0
        s = env.reset()
        # this will make the agent do an episode: move until 100 steps, goal or hole
        while True:
            # choose an action based on q we know until now
            a = self.action(s)
            # make the transition with a
            sf, r, is_done, _ = env.step(a)
            r_total += r
            if is_done:
                break
            s = sf
            p += 1
            #env.render()
        return p, r_total

    # get q matrix so that we can print it at the end
    def getQ(self):
        return self.q.getQ()

#=======================================================
# another run() but this one is to render it in the screen
def run(agent, env, render=False):
    p = 0
    done = False
    r_total = 0.0
    s = env.reset()
    while not done:
        a = agent.action(s)
        sf, r, done, info = env.step(a)
        s = sf
        r_total += r
        p += 1
        if (render):
            env.render()
    return p, r_total

def main():
    # choose the problem in gym: create two envs: one for train/learn and other for test
    env      = gym.make('FrozenLake-v0')
    env_test = gym.make('FrozenLake-v0')
    # get dimensions
    Ns = env.observation_space.n
    Na = env.action_space.n
    agent = Agent(Ns, Na)
    #print("Ns=",Ns,"Na=",Na)
    i = 0;
    r_max = 0.0
    s = env.reset()
    # each while iteration is a transition
    while True:
        # learn() does one random action and its transition in env and comes back, updating q
        s = agent.learn(s, env)

        #print("================================")
        r = 0.0
        episodes = 20
        # test the agent 20 times in env_test and compute avg reward to see if q is solved/good or not
        for _ in range(episodes):
            p, r_tmp = agent.run(env_test)
            r += r_tmp
            #print("p=",p," r=",r_tmp)
        r_prom = r/episodes
        if (r_prom > r_max):
            # i=321 means 321 transitions
            print("i=",i," r_prom=", r_max,"->", r_prom)
            r_max = r_prom
        #else:
            #print("i=",i," r_prom=", r_prom)
        if (r_prom > 0.8):
            print("Solved!")
            break
        i += 1

    # TEST: Solution
    # once the learning of optimal politic approximation is done, you can test the agent to see how cool it is
    # and with how many steps it solved a test (may fail due to stochastic world)
    test_agent = input('Test agent [1=yes|0=no]:')
    while (int(test_agent) == 1):
        p, r = run(agent, env, True)
        print("Steps=",p," r=", r)
        test_agent = input('Test agent [1=yes|0=no]:')
    env.close()
    #for s in range(Ns):
    #  for a in range(Na):
    #    print("s=",s," a=",a," p(sf|s,a)=",agente.p.p[s][a])
    #for s in range(Ns):
    #  print("s=",s," v(s)=",agente.v.v[s])
    return agent.getQ()

if __name__ == '__main__':
    q = main()
