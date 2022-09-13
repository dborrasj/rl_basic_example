############################# Librerías ##################################

import numpy as np
import matplotlib.pyplot as plt




####################### Definición del entorno ##########################


#Definición de las dimensiones del entorno (en este caso es espacial), 
#Entorno: matriz 11x11 con 121 estados posibles

environment_rows = 11
environment_columns = 11

#Inicializacion de la matriz de valores Q (q_values), las dos primeras 
#dimensiones definen cada uno de los estados y la tercera dimension cada una de las acciones
q_values = np.zeros ((environment_rows, environment_columns, 4))

#Definicion de acciones que puede ejecutar el algoritmo
actions = ["up", "right", "down",  "left"]


#Definicion de la matriz de recompensas recompensas
rewards = np.full((environment_rows, environment_columns), -100)#Se llena inicialmente la matriz de -100
rewards[0,5] = 100 #se reemplaza por un valor de 100 la posición 0,5

    #Se modifican las celdas donde se "puede transitar", path
    #primero, se crea un diccionario para almacenar las posiciones de las celdas transitables
dict_path_pos = {}
dict_path_pos[1] = [i for i in range (1,10)]
dict_path_pos[2] = [1,7,9]
dict_path_pos[3] = [i for i in range(1,8)]
dict_path_pos[3].append(9)
dict_path_pos[4] = [3,7]
dict_path_pos[5] = [i for i in range(11)]
dict_path_pos[6] = [5]
dict_path_pos[7] = [i for i in range(1,10)]
dict_path_pos[8] = [3,7]
dict_path_pos[9] = [i for i in range(11)]

    #se ponen los valores de la recompensa en las celdas transitables empleando la posicion de
    #estas (creadas en el diccionario)
for row in range(1,10):
    for column in dict_path_pos[row]:
        rewards[row,column]= -1.


##################### Proceso de aprendizaje ########################


################ Definición de Funciones 

#Definición de una funcion para especificar si el estado actual es terminal
def terminal_state_ver (current_row_index, current_column_index):
    #la verificación se hace teniendo en cuenta la recompensa de la celda o estado, si es -1, no 
    # es terminal
    if rewards[current_row_index, current_column_index] == -1.:
        return False 
    else:
        return True

#Definición de una función para seleccionar un punto de partida aleatorio y no terminal
def get_starting_location ():

    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)

    #while para cambiar el valor del index de la fila y columna hasta que no sea un estado terminal
    while terminal_state_ver(current_row_index, current_column_index):
        #se vuelve a obtener el index de fila y columna de manera aleatoria
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    
    return current_row_index, current_column_index

#Definición dl algoritmo Epsilon greedy, el cual define la acción siguiente
def get_next_action (current_row_index, current_column_index, epsilon):
    #Si aleatoriamente entre 0 y 1 se escoge un valor menos al epsilon,
    #se selecciona el valor mas prometedor de la tabla Q para este estado
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    # si no, se selecciona un valor aleatorio entre las acciones
    else:
        return np.random.randint(4)

#Definición de una función para obtener el siguiente estado teniendo en cuenta la accion seleccionada
def get_next_location (current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index

    if actions[action_index] == "up" and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == "right" and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == "down" and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == "left" and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index


#Definición de una función para determinar el camino mas corto entre la ubicación actual (estado) y 
# la locación de almacenamiento del item (estado objetivo)
def get_shortest_path(start_row_index, start_column_index):
  #retorno en caso de ser un estado inicial (ubiación) invalida
    if terminal_state_ver(start_row_index, start_column_index):
        return []
    
    else: #estado inicial valido
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
    #"movimiento" entre estados hasta llegar al estado objetivo
    while not terminal_state_ver(current_row_index, current_column_index):
        #determinar la mejor accion a tomar
        action_index = get_next_action(current_row_index, current_column_index, 1.) #epsilon de 1 para que siempre se vaya por el valor Q mas alto
        #se "mueve" al siguiente estado, y agrega el nuevo estado a la lista shortest_path
        current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
        shortest_path.append([current_row_index, current_column_index])#se agregan los indices de los estados
    return shortest_path

################ Entrenamiento con Q-learning

#Parametros de entrenamiento
epsilon = 0.9 #porcentaje de tiempo en el que el algoritmo seleccionara la "mejor opcion"
discount_factor = 0.9 #Factor de descuento para las recompensas futuras
learning_rate = 0.9 #"velocidad" a la que el agente aprende
iteraciones =700

#Proceso iterativo en "epocas" o episodios. Cada episodio termina cuando el agente se encuentra con un estado terminal o encuentra el objetivo

max_q_value_list = []
ave_q_value_list =[]
episodes = []


for episode in range(iteraciones):
    #Determinación del estado de inicio
    row_index, column_index = get_starting_location()

    while not terminal_state_ver(row_index, column_index):
        #Selección de la acción a tomar
        action_index = get_next_action(row_index, column_index, epsilon)

        #Ejecución de la acción y transición del siguiente estado
        old_row_index, old_column_index = row_index, column_index #almacenamiento de los indices del estado anteriores
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        #Recibimiento de la recompensa
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor*np.max(q_values[row_index, column_index]))- old_q_value


        #Actualizacion del Q-value del par estado-acción anterior
        new_q_value = old_q_value + (learning_rate*temporal_difference)
        q_values[old_row_index, old_column_index, action_index] = new_q_value
    print(episode)
    episodes.append(episode)
    max_q_value_list.append(np.max(q_values))
    ave_q_value_list.append(np.mean(q_values))
    
print("Aprendizaje terminado")

##################### Caso de aplicación ########################

#Posición inicial en fila 9, columna 10
print("El camino más optimo es: ", "\n", get_shortest_path(9 ,10))

print("\nTabla Q estado ({},{})".format(9,7), "\n", q_values[7,7,:],"\n",actions)

#Graficación de convergencia
fig, ax = plt.subplots(1,2)
ax[0].plot(episodes,max_q_value_list)
ax[1].plot(episodes,ave_q_value_list)
ax[0].set(title = "Maximum Q value", xlabel = "Episodes", ylabel ="Max Q value")
ax[1].set(title = "Average Q value", xlabel = "Episodes", ylabel ="Ave Q value")
plt.show()

