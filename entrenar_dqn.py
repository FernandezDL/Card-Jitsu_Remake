import numpy as np
from Qlearning import get_state, reward, inicializar_Q
from asignarCartas import cargar_cartas, asignar_cartas
import copia
from DeepQNetwork import DeepQLearningAgent
import time


# Definir TARGET_UPDATE para sincronizar la red de destino del DQN
TARGET_UPDATE = 10

def codificar_cadena(cadena):
    """Codifica una cadena específica a un valor numérico."""
    if cadena == 'Fuego':
        return 1
    elif cadena == 'Agua':
        return 2
    elif cadena == 'Nieve':
        return 3
    elif cadena == 'User':
        return 0
    elif cadena == 'IA':
        return 1
    elif cadena == 'Empate':
        return 2
    else:
        return -1  # Valor desconocido

def encode_state(state):
    """Codifica el estado y asegura longitud fija."""
    encoded_state = []
    for item in state:
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                if isinstance(sub_item, str):
                    encoded_state.append(codificar_cadena(sub_item))
                else:
                    encoded_state.append(sub_item)
        elif isinstance(item, str):
            encoded_state.append(codificar_cadena(item))
        else:
            encoded_state.append(item)
    
    # Asegurar que el estado tenga una longitud fija
    desired_length = 20  # Ajustar según sea necesario
    while len(encoded_state) < desired_length:
        encoded_state.append(0)  # Rellenar con ceros
    
    # Si es más largo, truncar al tamaño deseado
    if len(encoded_state) > desired_length:
        encoded_state = encoded_state[:desired_length]
    
    return encoded_state

def is_numeric_state(state):
    """Verifica que todos los elementos del estado sean numéricos."""
    for item in state:
        if not isinstance(item, (int, float)):
            return False
    return True

def entrenar_dqn_con_qlearning(episodios=1000):
    start_time = time.time()
    # Configuración inicial del modelo DQN
    state_dim = 20  # Longitud fija del estado
    action_dim = 5  # Suponiendo que siempre hay 5 cartas en la mano
    agent_dqn = DeepQLearningAgent(state_dim, action_dim)

    # Inicializar Q-Table solo para usarla, sin actualizarla
    inicializar_Q()

    for episodio in range(episodios):
        # Inicializar partida
        cartas = cargar_cartas()
        cartas_dqn, cartas_ql = asignar_cartas(cartas)
        mano_dqn, mazo_dqn = copia.seleccionar_cartas_mano(cartas_dqn)
        mano_ql, mazo_ql = copia.seleccionar_cartas_mano(cartas_ql)
        
        victorias = {"User": {"Fuego": [], "Agua": [], "Nieve": []}, "IA": {"Fuego": [], "Agua": [], "Nieve": []}}
        historial_acciones = []
        done = False

        while not done:
            # Obtener el estado actual
            estado_actual = get_state(victorias, mano_ql, mazo_dqn, mazo_ql, historial_acciones)
            estado_actual_encoded = encode_state(estado_actual)  # Codificar el estado para DQN
            # print(f"Longitud del estado codificado: {len(estado_actual_encoded)}")  # Opcional para depuración

            # Selección de acciones
            accion_dqn = agent_dqn.select_action(estado_actual_encoded)
            accion_ql = copia.select_action(estado_actual, mano_ql)  # El Q-Learning usa el estado sin codificar

            # Jugar las cartas seleccionadas
            carta_dqn = mano_dqn.pop(accion_dqn)
            carta_ql = mano_ql.pop(mano_ql.index(accion_ql))

            # Determinar el resultado del turno y la recompensa
            resultado = copia.determinar_ganador(carta_dqn, carta_ql, victorias)
            historial_acciones.append((carta_dqn.elemento, carta_ql.elemento, resultado))

            if resultado == "Empate":
                recompensa = reward(False, False)
            else:
                recompensa = reward(resultado == "User", resultado == "IA")

            # Obtener el estado siguiente
            estado_siguiente = get_state(victorias, mano_ql, mazo_dqn, mazo_ql, historial_acciones)
            estado_siguiente_encoded = encode_state(estado_siguiente)

            # Verificar que los estados sean numéricos antes de almacenarlos
            if is_numeric_state(estado_actual_encoded) and is_numeric_state(estado_siguiente_encoded):
                # Almacenar la experiencia y entrenar
                agent_dqn.store_experience(estado_actual_encoded, accion_dqn, recompensa, estado_siguiente_encoded, done)
                agent_dqn.train()
            else:
                print("Error: El estado contiene valores no numéricos:", estado_actual_encoded)

            # Verificar si hay un ganador
            ganador, victoria = copia.verificar_condicion_victoria(victorias)
            if ganador:
                if (episodio + 1) % 50 == 0:
                    print(f'Episodio {episodio + 1}: Ganador - {ganador} con {victoria}')
                done = True

            # Reemplazar cartas si quedan en el mazo
            if mazo_dqn and mazo_ql:
                nueva_carta_dqn = np.random.choice(mazo_dqn)
                mazo_dqn.remove(nueva_carta_dqn)
                mano_dqn.append(nueva_carta_dqn)

                nueva_carta_ql = np.random.choice(mazo_ql)
                mazo_ql.remove(nueva_carta_ql)
                mano_ql.append(nueva_carta_ql)

        # Actualizar red de destino cada TARGET_UPDATE episodios
        if (episodio + 1) % TARGET_UPDATE == 0:
            agent_dqn.update_target_network()

        # Mostrar progreso cada 50 episodios
        if (episodio + 1) % 50 == 0:
            print(f'Episodio {episodio + 1} completado.')

    # Medir el tiempo de finalización
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Entrenamiento completado en {total_time:.2f} segundos.")
    
    # Guardar el modelo DQN al finalizar el entrenamiento
    agent_dqn.save_model('dqn_trained_model.pth')
    print("Entrenamiento completado y modelo guardado en 'dqn_trained_model.pth'.")

# Ejecución de la función de entrenamiento
if __name__ == "__main__":
    entrenar_dqn_con_qlearning(episodios=1000)
