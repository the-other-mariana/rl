import gym

mundo = gym.make('CartPole-v0')
#mundo = gym.make('MountainCar-v0') #x∈[-1.2, 0.6], v∈[-0.07, 0.07], a∈{0(←),1,2(→)}, r=-1 en [-1.2,0.5]
#mundo = gym.make('BipedalWalker-v3')
#mundo = gym.make('SpaceInvaders-v0')
#mundo = gym.make('PongNoFrameskip-v4')
#mundo = gym.make('CarRacing-v0')
#mundo = gym.make('MsPacman-v0')
#mundo = gym.make('LunarLander-v2')
#mundo = gym.make('Qbert-v0')

print("mundo.observation_space.shape=", mundo.observation_space.shape)

espacio = mundo.observation_space
print("Espacio de Estados=", espacio)
if isinstance(espacio, gym.spaces.Box):
  print("xmin=", espacio.low)
  print("xMAX=", espacio.high)

espacio = mundo.action_space
print("Espacio de Acciones=", espacio)
if isinstance(espacio, gym.spaces.Box):
  print("xmin=", espacio.low)
  print("xMAX=", espacio.high)

mundo.reset()
for _ in range(1):
  mundo.render()
  accion = mundo.action_space.sample()
  sig_estado, recompensa, done, info = mundo.step(accion)
  print("acción=", accion, " sig_estado=", sig_estado)
  if (done):
    mundo.reset()
mundo.close()

'''
#MUNDOS: nombres
print("NOMBRE: ambientes");
env_names = [env.id for env in gym.envs.registry.all()]
for name in sorted(env_names):
  print("name=",name)
'''

'''
#ESPACIO: Continuo
# Box∈R^n, (x1,...,xn), xi∈[xmin, xMAX]
space = gym.spaces.Box(low=-10, high=10, shape=(2,)) #(x1,x2), -10<xi<10

#ESPACIO: Discreto
# Discrete∈{0,...,(n-1)}
space = gym.spaces.Discrete(5) #{0,1,2,3,4}

#ESPACIO: Diccionario de Espacios
space = gym.spaces.Dict({
          "x1": gym.spaces.Discrete(3), #{0,1,2}
          "x2": gym.spaces.Discrete(2), #{0,1}
          "x3": gym.spaces.Box(low=-10, high=10, shape=(2,)) #(x1,x2), -10<xi<10
        })

#ESPACIO: Multi-binario
# MultiBinary∈{t,f}^n, (x1,...,xn), xi∈{t,f}
space = gym.spaces.MultiBinary(3) #(x1,x2,x3), xi∈{t,f}

#ESPACIO: Multi-Discreto
# MultiDiscrete∈{a,a+1,...,b}^n
space = gym.spaces.MultiDiscrete([-10,10], [0,1])

#ESPACIO: Tupla=Producto cartesiano de espacios
# MultiDiscrete∈{a,a+1,...,b}^n
space = gym.spaces.Tuple(gym.spaces.Discrete(3),
                         gym.spaces.Discrete(2)) #{0,1,2}x{0,1}
'''
