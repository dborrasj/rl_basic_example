############################# Librerías ##################################

import numpy as np
import matplotlib.pyplot as plt
import gym
import random

#Librerias para Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Danse, Flatten
from tensorflow.keras.optimizers import Adam

####################### Definición del entorno ##########################






##################### Proceso de aprendizaje ########################

################ Definición de Funciones 


#Creacion de un modelo Deep Learning con Keras
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation = "relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))

