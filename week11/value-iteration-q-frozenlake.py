#RL: Value Iteration [Q]
#Q(s,a): Table
#PROBLEM: Frozen Lake
#API: -
# same code as before, only difference # CHANGE

import gym
import numpy as np

# states =0,1,...,15
# actions=0,1,2,3
γ = 0.9

#=======================================================
class P:
    def __init__(self, Ns, Na):
        self.f  = np.zeros((Ns, Na, Ns))
        self.p  = np.zeros((Ns, Na, Ns))

    def update(self, s, a, sf):
        self.f[s][a][sf] += 1
        #print("f(sf|s,a)=", self.f[s][a])
        Nsf = len(self.p[s][a])
        #print("Nsf=", Nsf)
        N = self.f[s][a].sum()
        if (N == 0):
            N = 1
        #print("N=", N)
        for i in range(Nsf):
            self.p[s][a][i] = self.f[s][a][i]/N
        #print("p(sf|s,a)=", self.p[s][a])

    def get(self, s, a, sf):
        return self.p[s][a][sf]

    # CHANGE: return P matrix
    def getP(self):
        return self.p

#=======================================================
class R:
    def __init__(self, Ns, Na):
        self.r = np.zeros((Ns, Na, Ns));

    def update(self, s, a, sf, r):
        self.r[s][a][sf] = r

    def get(self, s, a, sf):
        return self.r[s][a][sf]

    # CHANGE: return r matrix
    def getR(self):
        return self.r

#=======================================================
class Q:
    def __init__(self, Ns, Na):
        self.q  = np.zeros((Ns, Na))
        self.Ns = Ns
        self.Na = Na

    def update(self, s, a, q):
        self.q[s][a] = q

    def get(self, s, a):
        return self.q[s][a]

    def maximum(self, s):
        return max(self.q[s])

    def action(self, s):
        a_max = 0
        qs_max = -float('inf')
        for a in range(self.Na):
            if (self.q[s][a] > qs_max):
                a_max = a
                qs_max = self.q[s][a]
        return a_max

    # CHANGE: return q matrix
    def getQ(self):
        return self.q

#=======================================================
class Agent:
    def __init__(self, Ns, Na):
        self.Ns = Ns             #No. de estados
        self.Na = Na             #No. de acciones
        self.p  = P(Ns, Na)      #P(s'|s,a)
        self.r  = R(Ns, Na)      #R(s,a,s')
        self.q  = Q(Ns, Na)      #Q(s,a)

    def learn(self, env):
        # LEARNING: p(s'|s,a), r(s,a,s')
        s = env.reset()
        for p in range(100):
            a = env.action_space.sample()
            sf, r, is_done, _ = env.step(a)
            self.r.update(s, a, sf, r)
            self.p.update(s, a, sf)
            s = sf
            if (is_done):
                s = env.reset()

        #LEARNING: Bellman's Optimality Equations
        for s in range(self.Ns):
            for a in range(self.Na):
                qsa = 0
                for sf in range(self.Ns):
                    qsa += self.p.get(s,a,sf)*(self.r.get(s,a,sf) + γ*self.q.maximum(sf))
                self.q.update(s,a,qsa)

    def action(self, s):
        return self.q.action(s)

    def run(self, env):
        p = 0
        r_total = 0.0
        s = env.reset()
        while True:
            a = self.action(s)
            sf, r, is_done, _ = env.step(a)
            self.r.update(s, a, sf, r)    #NOTA: es clave actualizar aqui
            self.p.update(s, a, sf)       #NOTA: es clave actualizar aqui
            r_total += r
            if is_done:
                break
            s = sf
            p += 1
            #env.render()
        return p, r_total

    # CHANGE: since Agent creates P,R,Q objects, we return them
    def getPRQ(self):
        return self.p.getP(), self.r.getR(), self.q.getQ()

#=======================================================
    # print in the screen every state where the agent is moving
    def run(agent, env, render=False):
        p = 0
        done = False # episode not finished
        r_total = 0.0
        s = env.reset()
        # ONE EPISODE here (episode is from start to goal)
        while not done:
            # agent returns an action NOT RANDOM
            a = agent.action(s)
            # the env returns the reward of the transition given action a, state and info
            sf, r, done, info = env.step(a)
            s = sf
            r_total += r
            # steps +=1 (next transition)
            p += 1
            if (render):
                env.render()
        return p, r_total

def main():
    env = gym.make('FrozenLake-v0')
    Ns  = env.observation_space.n
    Na  = env.action_space.n
    # since main() creates the agent, the return PRQ is needed here
    agent = Agent(Ns, Na)
    #print("Ns=",Ns,"Na=",Na)
    i = 0;
    r_max = 0.0
    while True:
        agent.learn(env)
        #print("================================")
        r = 0.0
        episodes = 20
        for _ in range(episodes):
            p, r_tmp = agent.run(env)
            r += r_tmp
            #print("p=",p," r=",r_tmp)
        r_prom = r/episodes
        if (r_prom > r_max):
            print("i=",i," r_prom=", r_max,"->", r_prom)
            r_max = r_prom
        else:
            print("i=",i," r_prom=", r_prom)
        if (r_prom > 0.8):
            print("Solved!")
            break
        i += 1

    # CHANGE: once the learning/training is finished (eqs solved) test the agent now working
    # TEST: Solution
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
    return agent.getPRQ()

if __name__ == '__main__':
    # return prq here so that we can see them
    p, r, q = main()
