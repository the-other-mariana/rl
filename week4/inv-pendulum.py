#RL:  
#PROBLEMA: Pendulo Invertido
#API: OpenAI Gym

import gym

class Agente:
  def __init__(self):
    self.r = 0.0
    
  def accion(self, mundo: gym.Env):
    a = mundo.action_space.sample()
    s, r, fin, _ = mundo.step(a)
    self.r += r
    return fin

#PROGRAMA: Inicio
agente = Agente()
mundo = gym.make('CartPole-v0')    #Crear el mundo
#mundo = gym.wrappers.Monitor(mundo, "C:/Python/rec")
obs = mundo.reset()
fin = False
pasos = 0
while pasos <= 200: #not fin and pasos <= 100:
  mundo.render()
  fin = agente.accion(mundo)
  pasos += 1
  print(fin)
mundo.close()
r = agente.r
