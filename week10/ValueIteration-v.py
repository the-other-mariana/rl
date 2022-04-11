# RL: Value Iteration [V]
# V(s): Table
# PROBLEM: Frozen Lake: must arrive from S to G without falling into H
# API: -

import gym
import numpy as np

'''
SFFH
FFHF
HFFH
HHFG
'''
# S = start, H = hole (ends the game), G = goal (ends the game), F = free
# states =0,1,...,15, the 16 cells of the grid
# actions =0,1,2,3, meaning right, left, up, down
# action results are stochastic: Transition Model is stochastic
# the only state that rewards is G, which gives 1
γ = 0.9

#=======================================================
# probability distribution class: in this case is stochastic, and therefore 3D matrix
class P:
  def __init__(self, Ns, Na):
    # the transition model is not known, thus we calculate it
    self.f  = np.zeros((Ns, Na, Ns)) # identical to P: stores the frequency tables of a transition [s, a, sf]
    self.p  = np.zeros((Ns, Na, Ns)) # probability ditribution cube p[s, a, sf] = f[s, a, sf]/total transitions

  # calculate p tables
  def update(self, s, a, sf):
    # when a transition happens, we increase its f counter
    self.f[s][a][sf] += 1
    #print("f(sf|s,a)=", self.f[s][a])
    # get the number of final states given s and a
    Nsf = len(self.p[s][a])
    #print("Nsf=", Nsf)
    # get the total transitions given s and a: since p(sf|s,a), sum the total transitions for that s and a
    N = self.f[s][a].sum()
    if (N == 0):
      N = 1 # avoid zero division
    #print("N=", N)
    # calculate p over the 16 states (last dimension in p cube)
    for i in range(Nsf):
      # probability = the event / total events
      self.p[s][a][i] = self.f[s][a][i]/N
    #print("p(sf|s,a)=", self.p[s][a])

  # get the probability given this transition s, a and sf
  def get(self, s, a, sf):
    return self.p[s][a][sf]

#=======================================================
# reward function class
class R:
  def __init__(self, Ns, Na):
    # fr(s,a,sf) = matrix, where each cell is the 'fourth dimension'
    self.r = np.zeros((Ns, Na, Ns));

  # given the transition experimented and the reward it gave us, we store that
  def update(self, s, a, sf, r):
    self.r[s][a][sf] = r

  # get the reward value for a transition
  def get(self, s, a, sf):
    return self.r[s][a][sf]

#=======================================================
# V(s) function class
class V:
  def __init__(self, Ns):
    # 1D array
    self.v = np.zeros((Ns))

  def update(self, s, v):
    self.v[s] = v

  def get(self, s):
    return self.v[s]

#=======================================================
# Agent: the one who solves the equations
class Agent:
  def __init__(self, Ns, Na):
    self.Ns = Ns             #No. of states
    self.Na = Na             #No. de actions
    # build one object per class
    self.p  = P(Ns, Na)      # P(s'|s,a)
    self.r  = R(Ns, Na)      # R(s,a,s')
    self.v  = V(Ns)          # V(s)

  # here we solve the equations: value iteration method is done here
  # computes the final V(s) vector
  def learn(self, env):
    #LEARNING: p(s'|s,a), r(s,a,s')
    s = env.reset() # get an initial state
    # learn the transition model and the reward function by trying 100 random actions (transitions)
    for p in range(100):
      a = env.action_space.sample() # get a random action
      sf, r, is_done, _ = env.step(a) # do the transition
      self.r.update(s, a, sf, r)
      self.p.update(s, a, sf)
      s = sf
      if (is_done):
        s = env.reset() # if we fall on an H or G, we start again

    #LEARNING: Bellman's Optimality Equations for V(s), once we have p and r
    # eq = max (term1 = sum of 16 (probabilities), term2, term3, term4) since 4 actions
    # here is the Value Iteration Method
    # V(s) is size 16 (states), thus we approximate 16 terms here
    for s in range(self.Ns):
      # each V(s) is approximated here
      vsa = np.zeros((self.Na))
      # we have Na terms which are a sum of 16 terms
      for a in range(self.Na):
        # these two lines do the term = sum of 16, vsa is size 4
        for sf in range(self.Ns):
          vsa[a] += self.p.get(s,a,sf)*(self.r.get(s,a,sf) + γ*self.v.get(sf))
      # get max of 4 terms
      vsa_max = max(vsa)
      # store in v the approximation
      self.v.update(s, vsa_max)

  # the agent takes an action based on what is learned from V(s): deduce the action by searching which gave us the max
  def action(self, s):
    a_max   = 0
    vsa_max = -float('inf')
    vsa     = np.zeros((self.Na))
    # get the action that gaves us the max value
    for a in range(self.Na):
      for sf in range(self.Ns):
        vsa[a] += self.p.get(s,a,sf)*(self.r.get(s,a,sf) + γ*self.v.get(sf))
      if (vsa[a] > vsa_max):
        a_max = a
        vsa_max = vsa[a]
    return a_max

  # move the agent
  def run(self, env):
    p = 0 # total steps
    r_total = 0.0 # sum of rewards, not accumulated (no gamma)
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
      #env.render() # see the world on screen
    return p, r_total

#=======================================================
def main():
  env = gym.make('FrozenLake-v0') # create the world
  Ns  = env.observation_space.n # get n of states
  Na  = env.action_space.n # get n of actions
  agente = Agent(Ns, Na)
  print("Ns=",Ns,"Na=",Na)
  i = 0
  r_max = -float('inf')
  while True:
    agente.learn(env)

    #print("================================")
    # calculate episode stats
    r = 0.0
    # an episode is a cycle that ends with H or G
    episodios = 20
    # before another iteration, we test on 20 episodes
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
