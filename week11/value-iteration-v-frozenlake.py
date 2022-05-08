#RL: IteraciÃ³n de Valor [V]
#V(s): Tabla
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
    self.f  = np.zeros((Ns, Na, Ns))   #f=frecuencia
    self.p  = np.zeros((Ns, Na, Ns))   #p=probabilidad

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

  def obtenerP(self):
    return self.p

#=======================================================
class R:
  def __init__(self, Ns, Na):
    self.r = np.zeros((Ns, Na, Ns));
  
  def actualizar(self, s, a, sf, r):
    self.r[s][a][sf] = r
  
  def obtener(self, s, a, sf):
    return self.r[s][a][sf]

  def obtenerR(self):
    return self.r

#=======================================================
class V:
  def __init__(self, Ns):
    self.v = np.zeros((Ns))
  
  def actualizar(self, s, v):
    self.v[s] = v

  def obtener(self, s):
    return self.v[s]

  def obtenerV(self):
    return self.v

#=======================================================
class Agente:
  def __init__(self, Ns, Na):
    self.Ns = Ns             #No. de estados
    self.Na = Na             #No. de acciones
    self.p  = P(Ns, Na)      #P(sf|s,a)
    self.r  = R(Ns, Na)      #R(s,a,sf)
    self.v  = V(Ns)          #V(s)
    
  def aprender(self, env):
    #APRENDIZAJE: p(s'|s,a), r(s,a,s')
    s = env.reset()
    for p in range(100):
      a = env.action_space.sample()
      sf, r, is_done, _ = env.step(a)
      self.r.actualizar(s, a, sf, r)
      self.p.actualizar(s, a, sf)
      s = sf
      if (is_done):
        s = env.reset()

    #APRENDIZAJE: Ecuaciones de Optimalidad de Bellman
    for s in range(self.Ns):
      vsa = np.zeros((self.Na))
      for a in range(self.Na):
        for sf in range(self.Ns):
          vsa[a] += self.p.obtener(s,a,sf)*(self.r.obtener(s,a,sf) + γ*self.v.obtener(sf))
      vsa_max = max(vsa)
      self.v.actualizar(s, vsa_max)
  
  def accion(self, s):
    a_max   = 0
    vsa_max = -float('inf')
    vsa     = np.zeros((self.Na))
    for a in range(self.Na):
      for sf in range(self.Ns):
        vsa[a] += self.p.obtener(s,a,sf)*(self.r.obtener(s,a,sf) + γ*self.v.obtener(sf))
      if (vsa[a] > vsa_max):
        a_max = a
        vsa_max = vsa[a]
    return a_max
  
  def run(self, env):
    p = 0
    r_total = 0.0
    s = env.reset()
    while True:
      a = self.accion(s)
      sf, r, is_done, _ = env.step(a)
      self.r.actualizar(s, a, sf, r)    #NOTA: es clave actualizar aqui 
      self.p.actualizar(s, a, sf)       #NOTA: es clave actualizar aqui
      r_total += r
      if is_done:
        break
      s = sf
      p += 1
      #env.render()
    return p, r_total

  def obtenerPRV(self):
    return self.p.obtenerP(), self.r.obtenerR(), self.v.obtenerV()

#=======================================================
def run(agente, env, render=False):
  p = 0
  done = False
  r_total = 0.0
  s = env.reset()
  while not done:
    a = agente.accion(s)
    sf, r, done, info = env.step(a)
    s = sf
    r_total += r
    p += 1
    if (render):
      env.render()
  return p, r_total

def main():
  env = gym.make('FrozenLake-v0')
  Ns  = env.observation_space.n
  Na  = env.action_space.n
  agente = Agente(Ns, Na)
  print("Ns=",Ns,"Na=",Na)
  i = 0;
  r_max = -float('inf')
  print("r_max=",-float('inf'))
  while True:
    agente.aprender(env)

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
      print("MUNDO: Resuelto")
      break
    i += 1

  #PROBAR: Solución
  probar_agente = input('Probar agente [1=si|0=no]:')
  while (int(probar_agente) == 1):
    p, r = run(agente, env, True)
    print("Pasos=",p," r=", r)
    probar_agente = input('Probar agente [1=si|0=no]:')
  env.close()
  #for s in range(Ns):
  #  for a in range(Na):
  #    print("s=",s," a=",a," p(sf|s,a)=",agente.p.p[s][a])
  #for s in range(Ns):
  #  print("s=",s," v(s)=",agente.v.v[s])
  return agente.obtenerPRV()

if __name__ == '__main__':
  p, r, v = main()