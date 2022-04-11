#RL: Iteración de Valor [Q]
#Q(s,a): Tabla
#PROBLEMA: Frozen Lake
#API: -

import gym
import numpy as np

#estados =0,1,...,15
#acciones=0,1,2,3
γ = 0.9

#=======================================================
class P:
  def __init__(self, Ns, Na):
    self.f  = np.zeros((Ns, Na, Ns))
    self.p  = np.zeros((Ns, Na, Ns))

  def actualizar(self, s, a, sf):
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

  def obtener(self, s, a, sf):
    return self.p[s][a][sf]

#=======================================================
class R:
  def __init__(self, Ns, Na):
    self.r = np.zeros((Ns, Na, Ns));

  def actualizar(self, s, a, sf, r):
    self.r[s][a][sf] = r

  def obtener(self, s, a, sf):
    return self.r[s][a][sf]

#=======================================================
# Q(s,a) function class
class Q:
  def __init__(self, Ns, Na):
    # create a 2D table
    self.q  = np.zeros((Ns, Na))
    self.Ns = Ns
    self.Na = Na

  # store in q a value for transition s,a
  def update(self, s, a, q):
    self.q[s][a] = q

  def get(self, s, a):
    return self.q[s][a]

  # max of q[s] vector: it has the size of the amount of actions
  def maximum(self, s):
    return max(self.q[s])

  # returns the action with the maximum value: build the politic
  def action(self, s):
    a_max = 0 # starts assumming that a0 is the one with the maximum
    qs_max = -float('inf') # init for max search
    # max search
    for a in range(self.Na):
      if (self.q[s][a] > qs_max):
        a_max = a
        qs_max = self.q[s][a]
    return a_max

#=======================================================
# agent class: same as V, only changes the Bellman's Equations, which are now for Q
class Agent:
  def __init__(self, Ns, Na):
    self.Ns = Ns             #No. of states
    self.Na = Na             #No. of actions
    self.p  = P(Ns, Na)      # P(s'|s,a)
    self.r  = R(Ns, Na)      # R(s,a,s')
    self.q  = Q(Ns, Na)      # Q(s,a)

  def learn(self, env):
    #LEARNING: p(s'|s,a), r(s,a,s')
    # experiment the world 100 times, to update R and Ps
    s = env.reset()
    for p in range(100):
      a = env.action_space.sample()
      sf, r, is_done, _ = env.step(a)
      self.r.update(s, a, sf, r)
      self.p.update(s, a, sf)
      s = sf
      if (is_done):
        s = env.reset()

    #LEARNING: Bellman's Optimality Equations for Q(s,a), using Value Iteration
    # for each state
    for s in range(self.Ns):
      # calculate 4 sums of 16 terms, but we store them instead of taking max of the 4 terms (sums)
      for a in range(self.Na):
        qsa = 0
        # the sum of 16 terms done here
        for sf in range(self.Ns):
          qsa += self.p.get(s,a,sf)*(self.r.get(s,a,sf) + γ*self.q.maximum(sf))
        self.q.update(s,a,qsa) # update Q(s,a) = qsa, for that a, instead of maximum as with V(s)

  # the agent decides on the action by asking q table which action a gives the maximum q for that state
  def action(self, s):
    return self.q.action(s)

  # same as V(s) version
  def run(self, env):
    p = 0
    r_total = 0.0
    s = env.reset()
    while True:
      a = self.action(s)
      sf, r, is_done, _ = env.step(a)
      self.r.update(s, a, sf, r)    #NOTE: key to update here, keeps learning
      self.p.update(s, a, sf)       #NOTE: key to update here, keeps learning
      r_total += r
      if is_done:
        break
      s = sf
      p += 1
      #env.render()
    return p, r_total

#=======================================================
def main():
  env = gym.make('FrozenLake-v0')
  Ns  = env.observation_space.n
  Na  = env.action_space.n
  agente = Agent(Ns, Na)
  #print("Ns=",Ns,"Na=",Na)
  i = 0;
  r_max = 0.0
  while True:
    agente.learn(env)

    #print("================================")
    r = 0.0
    episodios = 20
    for _ in range(episodios):
      p, r_tmp = agente.run(env)
      r += r_tmp
      #print("p=",p," r=",r_tmp)
    r_prom = r/episodios
    if (r_prom > r_max):
      print("i=",i," r_prom=", r_max,"->", r_prom)
      r_max = r_prom
    else:
      print("i=",i," r_prom=", r_prom)
    if (r_prom > 0.8):
      print("¡Resuelto!")
      break
    i += 1
  env.close()
  #for s in range(Ns):
  #  for a in range(Na):
  #    print("s=",s," a=",a," p(sf|s,a)=",agente.p.p[s][a])
  #for s in range(Ns):
  #  print("s=",s," v(s)=",agente.v.v[s])

if __name__ == '__main__':
  main()
