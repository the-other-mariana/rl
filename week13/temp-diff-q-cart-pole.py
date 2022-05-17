#RL: Q-Learning [Q] usando TD=Diferencia Temporal
#Q(s,a): Tabla
#PROBLEMA: Montain Car
#API: -

import gym
import numpy as np

#recompensa: r=-1 en todo s=(x,v)
MAX_NUM_EPISODIOS = 100000
PASOS_POR_EPISODIO = 200
max_num_pasos = MAX_NUM_EPISODIOS * PASOS_POR_EPISODIO
MUNDO = "CartPole-v0"
ε_MAX = 1
ε_min = 0.005
dε = ε_MAX / (MAX_NUM_EPISODIOS*3) # the slope with which the curves of prob lower or increase
α = 0.05        #Razón de aprendizaje
γ = 0.98        #Factor de descuento
# one point if the action didnt lose the pendulum

np.random.seed(0)

#=======================================================
class Q:
  def __init__(self, env):
    #estados : x∈[-4.8, 4.8], v∈[-3.4x10+38, 3.4x10+38], Θ∈[-0.4, 0.4], ω∈[-3.4x10+38, 3.4x10+38]
    #acciones: a∈{0(←),1(→)}
    self.Nx = 30                                           #   No. de divisiones de x=posición:  x∈[-4.8, 4.8]
    self.Nv = 30                                           #   No. de divisiones de v=velocidad: v∈[-3.4x10+38, 3.4x10+38]
    self.NΘ = 30                                           #   No. de divisiones de x=posición:  Θ∈[-0.4, 0.4]
    self.Nω = 30                                           #   No. de divisiones de v=velocidad: ω∈[-3.4x10+38, 3.4x10+38]
    self.Ns = env.observation_space.shape                  #Ns=No. de variables de estado=2: s=(x,v,Θ,ω), x∈[-4.8, 4.8], v∈[-3.4x10+38, 3.4x10+38], Θ∈[-0.4, 0.4], ω∈[-3.4x10+38, 3.4x10+38]
    self.Na = env.action_space.n                           #   No. de variables de acción=1: Na=No. de acciones=2: a∈{0(←),1(→)}
    self.q  = np.zeros((self.Nx+1, self.Nv+1, self.NΘ+1, self.Nω+1, self.Na))      #Tabla: Q(s=(x,v,Θ,ω),a)
	# UNCOMMENT FOR EXPLOITATION ALWAYS (NO EXPLOARTION, NEVER LEARNS) THIS DEPENDS ON THE PROBLEM, MAY CHANGE
    #self.q  = np.random.rand(self.Nx+1, self.Nv+1, self.NΘ+1, self.Nω+1, self.Na)  #Tabla: Q(s=(x,v,Θ,ω),a)
    self.xmin = -4.8
    self.xMAX =  4.8
    self.vmin = -20#-3.4e38
    self.vMAX =  20# 3.4e38
    self.Θmin = -0.4
    self.ΘMAX =  0.4
    self.ωmin = -20#-3.4e38
    self.ωMAX =  20# 3.4e38
    self.dx = (self.xMAX - self.xmin)/self.Nx
    self.dv = (self.vMAX - self.vmin)/self.Nv
    self.dΘ = (self.ΘMAX - self.Θmin)/self.NΘ
    self.dω = (self.ωMAX - self.ωmin)/self.Nω

  # get the int, int coordinates inside matrix instead of float, float
  def discretizar(self, s):
    x = np.int32(np.floor((s[0]-self.xmin)/self.dx))
    v = np.int32(np.floor((s[1]-self.vmin)/self.dv))
    Θ = np.int32(np.floor((s[2]-self.Θmin)/self.dΘ))
    ω = np.int32(np.floor((s[3]-self.ωmin)/self.dω))
    return x, v, Θ, ω

  def actualizar(self, s, a, q):
    x, v, Θ, ω = self.discretizar(s)
    #print("s=",s,"===> x=",x," v=",v," Θ=",Θ," ω=",ω)
    self.q[x][v][Θ][ω][a] = q

  def obtener_(self, s):
    x, v, Θ, ω = self.discretizar(s)
    #print("x=",x," v=",v," Θ=",Θ," ω=",ω)
    #print("q=",self.q[x][v][Θ][ω])
    #qs = [self.q[x][v][Θ][ω][0], self.q[x][v][Θ][ω][1]]
    return self.q[x][v][Θ][ω]

  def obtener(self, s, a):
    x, v, Θ, ω = self.discretizar(s)
    return self.q[x][v][Θ][ω][a]

  def maximo(self, s):
    x, v, Θ, ω = self.discretizar(s)
    return max(self.q[x][v][Θ][ω])

  def accion(self, s):
    a_max = 0
    qs_max = -float('inf')
    x, v, Θ, ω = self.discretizar(s)
    for a in range(self.Na):
      if (self.q[x][v][Θ][ω][a] > qs_max):
        a_max = a
        qs_max = self.q[x][v][Θ][ω][a]
    return a_max

  def obtenerQ(self):
    return self.q

#=======================================================
class Agente:
  def __init__(self, env):
    self.q = Q(env)   #Q(s=(x,v),a)
    self.ε = ε_MAX

  def aprender(self, s, a, r, sf):
    #APRENDIZAJE: Ecuaciones de Optimalidad de Bellman
    qsa = self.q.obtener(s,a) + α*((r + γ*self.q.maximo(sf)) - self.q.obtener(s,a))
    self.q.actualizar(s,a,qsa)

  def accion(self, s, env):
	# UNCOMMENT FOR EXPLOITATION ALWAYS (NO EXPLOITATION)
    #return self.q.accion(s)               #EXPLOTACIÓN: a=maxQ(s,a)
	# decrease the slope of exploration, decrease the current probability
	# decide how to take action: random or with q
    if (self.ε > ε_min):
      self.ε -= dε
	# this is the thing that flips a coin and decides with current prob
    if (np.random.random() < self.ε): # eps is the prob of explr
      return np.random.choice([0, 1])       #EXPLORACIÓN: Accion al azar
    else:
      return self.q.accion(s)               #EXPLOTACIÓN: a=maxQ(s,a)

  def obtenerQ(self):
    return self.q.obtenerQ()

#=======================================================
def run(agente, env, render=False):
  p = 0
  done = False
  r_total = 0.0
  s = env.reset()
  while not done:
    a = agente.accion(s, env)
    sf, r, done, info = env.step(a)
    s = sf
    r_total += r
    p += 1
    if (render):
      env.render()
  return p, r_total

def probar(agente, env):
  Nepisodios = 20
  r = 0
  for _ in range(Nepisodios):
    p, r_total = run(agente, env)
    r += r_total
    #print("p=",p," r=",r)
  r_prom = r/Nepisodios
  return r_prom

def entrenar(agente, env):
  f = open('A.txt','w')
  r_MAX = -float('inf')
  r_prom = -float('inf')
  for e in range(10000):
    done = False
    s = env.reset()
    r_total = 0.0
    while not done:
      a = agente.accion(s, env)
      sf, r, done, info = env.step(a)
      #qs1 = agente.q.obtener_(s)
      #print("s=",s)
      #qsa1 = agente.q.obtener(s,a)
      #print(" qs1=", qs1)
      agente.aprender(s, a, r, sf)
      #qs2 = agente.q.obtener_(s)
      #print("s=",s)
      #qsa2 = agente.q.obtener(s,a)
      #print(" qs1=", qs1)
      #print(" qs2=", qs2)
      #print(" qsa1=", qsa1)
      #print(" qsa2=", qsa2)
      #print("s=",s," a=",a," r=",r," sf=",sf)
      #print("a=",a," q(s,a)=",qsa1,"==>",qsa2," q(s)=",qs1,"==>",qs2," s=",s," r=",r," sf=",sf,"\n")
      #f.write("a="+str(a)+" q(s,a)="+str(qsa1)+"==>"+str(qsa2)+" q(s)="+str(qs1)+"==>"+str(qs2)+" s="+str(s)+" r="+str(r)+" sf="+str(sf)+"\n")
      s = sf
      r_total += r
    if r_total > r_MAX:
      r_MAX = r_total
    resuelto = r_total > 195.0
    if (e%1000 == 0):
      r_prom = probar(agente, env)
    print("e=", e," r_total=", r_total," r_MAX=", r_MAX," r_prom=", r_prom," epsilon=", agente.ε, " resuleto=", resuelto)
    f.write("e="+str(e)+" r_total="+str(r_total)+" r_MAX="+str(r_MAX)+" r_prom="+str(r_prom)+" epsilon="+str(agente.ε)+" resuleto="+str(resuelto)+"\n")
  f.close()

def main():
  env = gym.make(MUNDO)
  env.seed(0)
  agente = Agente(env)
  entrenar(agente, env)

  #env.close()
  #return

  #PROBAR: Solución
  probar_agente = input('Probar agente [1=si|0=no]:')
  while (int(probar_agente) == 1):
    p, r_total = run(agente, env, True)
    print("p=", p," r_total=", r_total)
    probar_agente = input('Probar agente [1=si|0=no]:')
  env.close()
  return agente.obtenerQ()

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
