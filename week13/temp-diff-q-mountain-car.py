#RL: Q-Learning [Q] usando TD=Diferencia Temporal
#Q(s,a): Tabla
#PROBLEMA: Montain Car
#API: -

import gym
import numpy as np

#recompensa: r=-1 en todo s=(x,v)
MAX_NUM_EPISODES = 10000 # determines if the agent could learn a good politic when tested
PASOS_POR_EPISODIO = 200
WORLD = "MountainCar-v0"
ε_MAX = 1
ε_min = 0.005
dε = ε_MAX / (MAX_NUM_EPISODES*10)
α = 0.05        #Razón de aprendizaje
γ = 0.98        #Factor de descuento

np.random.seed(0)

#=======================================================
class Q:
	def __init__(self, env):
		#states : x∈[-1.2, 0.6], v∈[-0.07, 0.07]
		#actions: a∈{0(←),1,2(→)}
		self.Nx = 30                                            #   No. of divisions of x positions:  x∈[-1.2, 0.6]
		self.Nv = 30                                            #   No. of divisions of v velocities: v∈[-0.07, 0.07]
		self.Ns = env.observation_space.shape                   #	Ns=No. of state variables=2: s=(x,v), x∈[-1.2, 0.6], v∈[-0.07, 0.07]
		self.Na = env.action_space.n                            #   No. of action variables=1: Na=No. of actions=3: a∈{0(←),1,2(→)}
		#self.q  = np.zeros((self.Nx+1, self.Nv+1, self.Na))    # Table: Q(s=(x,v),a)
		# table Q(s,a) initially with random numbers [0,1]: starts with random actions
		self.q  = np.random.rand(self.Nx+1, self.Nv+1, self.Na) # Table: Q(s=(x,v),a)
		self.xmin = -1.2
		self.xMAX =  0.6
		self.vmin = -0.07
		self.vMAX =  0.07
		self.dx = (self.xMAX - self.xmin)/self.Nx
		self.dv = (self.vMAX - self.vmin)/self.Nv

	
	def discretize(self, s):
		x = np.int32(np.floor((s[0]-self.xmin)/self.dx))
		v = np.int32(np.floor((s[1]-self.vmin)/self.dv))
		return x, v

	def update(self, s, a, q):
		x, v = self.discretize(s)
		#print("s=",s,"===> x=", x," v=",v)
		self.q[x][v][a] = q

	def get_(self, s):
	    x, v = self.discretize(s)
	    return self.q[x][v]

	def get(self, s, a):
		x, v = self.discretize(s)
		return self.q[x][v][a]

	def maximum(self, s):
		x, v = self.discretize(s)
		return max(self.q[x][v])

	def action(self, s):
		a_max = 0
		qs_max = -float('inf')
		x, v = self.discretize(s)
		for a in range(self.Na):
		  if (self.q[x][v][a] > qs_max):
		    a_max = a
		    qs_max = self.q[x][v][a]
		return a_max

	def getQ(self):
		return self.q

#=======================================================
class Agent:
	def __init__(self, env):
		self.q = Q(env)   #Q(s=(x,v),a)
		self.ε = ε_MAX

	def learn(self, s, a, r, sf):
		#LEARNING: Bellman Optimality Equations
		qsa = self.q.get(s,a) + α*((r + γ*self.q.maximum(sf)) - self.q.get(s,a))
		self.q.update(s,a,qsa)

	def action(self, s, env):
		return self.q.accion(s)               #EXPLOITATION: a=maxQ(s,a)
		#if (self.ε > ε_min):
		#  self.ε -= dε
		#if (np.random.random() < self.ε):
		#  return np.random.choice([0, 1, 2])    #EXPLORATION: Accion al azar
		#else:
		#  return self.q.accion(s)               #EXPLOITATION: a=maxQ(s,a)

	def getQ(self):
		return self.q.getQ()

#=======================================================
def run(agent, env, render=False):
	p = 0
	done = False
	r_total = 0.0
	s = env.reset()
	while not done:
		a = agent.action(s, env)
		sf, r, done, info = env.step(a)
		s = sf
		r_total += r
		p += 1
		if (render):
			env.render()
	return p, r_total

def test(agent, env):
	Nepisodes = 20
	r = 0
	for _ in range(Nepisodes):
		p, r_total = run(agent, env)
		r += r_total
	#print("p=",p," r=",r)
	r_prom = r/Nepisodes
	return r_prom

def train(agent, env):
	f = open('A.txt','w')
	r_MAX = -float('inf')
	r_prom = -float('inf')
	for e in range(MAX_NUM_EPISODES):
		done = False
		s = env.reset()
		r_total = 0.0
	while not done:
		a = agent.action(s, env)
		sf, r, done, info = env.step(a)
		#qs1 = agente.q.obtener_(s)
		#qsa1 = agente.q.obtener(s,a)
		agent.learn(s, a, r, sf)
		#qs2 = agente.q.obtener_(s)
		#qsa2 = agente.q.obtener(s,a)
		#print("s=",s," a=",a," r=",r," sf=",sf)
		#f.write("a="+str(a)+" q(s,a)="+str(qsa1)+"==>"+str(qsa2)+" q(s)="+str(qs1)+"==>"+str(qs2)+" s="+str(s)+" r="+str(r)+" sf="+str(sf)+"\n")
		s = sf
		r_total += r
		if r_total > r_MAX:
			r_MAX = r_total
	    if (e%1000 == 0):
			r_prom = test(agent, env)
		print("e=", e," r_total=", r_total," r_MAX=", r_MAX," r_prom=", r_prom," epsilon=", agente.ε)
		f.write("e="+str(e)+" r_total="+str(r_total)+" r_MAX="+str(r_MAX)+" r_prom="+str(r_prom)+" epsilon="+str(agente.ε)+"\n")
	f.close()

def main():
	env = gym.make(WORLD)
	env.seed(0)
	agent = Agent(env)
	entrenar(agent, env)

	#TEST: solution
	test_agent = input('Test agent [1=yes|0=no]:')
	while (int(test_agent) == 1):
		p, r_total = run(agent, env, True)
		print("p=", p," r_total=", r_total)
		test_agent = input('Test agent [1=yes|0=no]:')
	env.close()
	return agent.getQ()

if __name__ == '__main__':
	q = main()

'''
#CODIGO: Prueba
env = gym.make(MUNDO)
x = [0, 1, 2]
print("x=", x)
for _ in range(10):
  n = np.random.choice([0, 1, 2])
  print("n=",n)
for _ in range(10):
  n = np.random.random()
  print("n=",n)

q = Q(env)
s = [-0.85, -0.052]
sf = [-1, -0.05]
xv = q.discretizar(s)
xvf = q.discretizar(sf)
a = 2
qv = 20
print("s=",s,"===> xv=",xv)
print("sf=",sf,"===> xvf=",xvf)
q.actualizar(s, 0, 3)
q.actualizar(s, 1, 5)
q.actualizar(s, 2, 2)
print("q(s, a)=",q.obtener(s, a))
print("max q(s)=", q.maximo(s))
print("a(s)=", q.accion(s))
t = q.q

agente = Agente(env)
agente.q.actualizar(s, 0, 3)
agente.q.actualizar(s, 1, 5)
agente.q.actualizar(s, 2, 2)
agente.q.actualizar(sf, 0, 1)
agente.q.actualizar(sf, 1, 4)
agente.q.actualizar(sf, 2, 6)
a = 1
r = -1
agente.aprender(s, a, r, sf)
print("q(s, a)=",agente.q.obtener(s, a))
t = agente.q.q
print("a(s)=", agente.accion(s, env))
print("a(sf)=", agente.accion(sf, env))
'''
