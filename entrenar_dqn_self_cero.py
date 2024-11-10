import time  # Importar el módulo time para medir el tiempo
import torch
import numpy as np
from Qlearning import get_state, reward
from asignarCartas import cargar_cartas
import copia
from DeepQNetwork import DeepQLearningAgent

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

def entrenar_dqn_selfplay_desde_cero(episodios=10000):
    # Medir el tiempo de inicio
    start_time = time.time()
    
    # Configuración inicial del modelo DQN
    state_dim = 20  # Longitud fija del estado
    action_dim = 5  # Suponiendo que siempre hay 5 cartas en la mano
    agent_dqn = DeepQLearningAgent(state_dim, action_dim)      # Agente principal (User)
    opponent_dqn = DeepQLearningAgent(state_dim, action_dim)   # Agente oponente (IA)
    
    # No cargar ningún modelo pre-entrenado; ambos agentes inician desde cero

    for episodio in range(episodios):
        # Inicializar partida
        cartas = cargar_cartas()
        cartas_user, cartas_ia = copia.asignar_cartas(cartas)
        mano_user, mazo_user = copia.seleccionar_cartas_mano(cartas_user)
        mano_ia, mazo_ia = copia.seleccionar_cartas_mano(cartas_ia)
        
        victorias = {"User": {"Fuego": [], "Agua": [], "Nieve": []}, "IA": {"Fuego": [], "Agua": [], "Nieve": []}}
        historial_acciones = []
        done = False

        while not done:
            # Obtener el estado actual
            estado_actual = get_state(victorias, mano_ia, mazo_user, mazo_ia, historial_acciones)
            estado_actual_encoded = encode_state(estado_actual)  # Codificar el estado para DQN

            # Selección de acciones
            accion_user = agent_dqn.select_action(estado_actual_encoded)
            accion_ia = opponent_dqn.select_action(estado_actual_encoded)

            # Jugar las cartas seleccionadas
            # Verificar que las acciones sean válidas (dentro del rango de la mano)
            accion_user = min(accion_user, len(mano_user) - 1)
            accion_ia = min(accion_ia, len(mano_ia) - 1)
            carta_user = mano_user.pop(accion_user)
            carta_ia = mano_ia.pop(accion_ia)

            # Determinar el resultado del turno y la recompensa
            resultado = copia.determinar_ganador(carta_user, carta_ia, victorias)
            historial_acciones.append((carta_user.elemento, carta_ia.elemento, resultado))

            if resultado == "Empate":
                recompensa = reward(False, False)
            else:
                recompensa = reward(resultado == "User", resultado == "IA")

            # Obtener el estado siguiente
            estado_siguiente = get_state(victorias, mano_ia, mazo_user, mazo_ia, historial_acciones)
            estado_siguiente_encoded = encode_state(estado_siguiente)

            # Verificar que los estados sean numéricos antes de almacenarlos
            if is_numeric_state(estado_actual_encoded) and is_numeric_state(estado_siguiente_encoded):
                # Almacenar la experiencia y entrenar al agente principal
                agent_dqn.store_experience(estado_actual_encoded, accion_user, recompensa, estado_siguiente_encoded, done)
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
            if mazo_user and mazo_ia:
                nueva_carta_user = np.random.choice(mazo_user)
                mazo_user.remove(nueva_carta_user)
                mano_user.append(nueva_carta_user)

                nueva_carta_ia = np.random.choice(mazo_ia)
                mazo_ia.remove(nueva_carta_ia)
                mano_ia.append(nueva_carta_ia)

        # Actualizar red de destino cada TARGET_UPDATE episodios
        if (episodio + 1) % TARGET_UPDATE == 0:
            agent_dqn.update_target_network()

        # Sincronizar el agente oponente con el agente principal cada 100 episodios
        if (episodio + 1) % 100 == 0:
            opponent_dqn.policy_net.load_state_dict(agent_dqn.policy_net.state_dict())
            print(f"Episodio {episodio + 1}: Sincronizando el agente oponente con el agente principal.")

        # Mostrar progreso cada 50 episodios
        if (episodio + 1) % 50 == 0:
            print(f'Episodio {episodio + 1} completado.')

    # Medir el tiempo de finalización
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Entrenamiento completado en {total_time:.2f} segundos.")

    # Guardar el modelo DQN al finalizar el entrenamiento
    agent_dqn.save_model('models_self/dqn_trained_selfplay_cero_15000.pth')
    print("Modelo guardado en 'dqn_trained_selfplay_desde_cero.pth'.")

# Ejecución de la función de entrenamiento
if __name__ == "__main__":
    entrenar_dqn_selfplay_desde_cero(episodios=15000)
