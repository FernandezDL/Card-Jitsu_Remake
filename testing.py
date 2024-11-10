import torch
import numpy as np
from Qlearning import get_state, reward, inicializar_Q
from asignarCartas import cargar_cartas, asignar_cartas
from DeepQNetwork import DeepQLearningAgent
import copia
from entrenar_dqn import encode_state

def evaluar_modelo_dqn_vs_qlearning(modelo_dqn, path, episodios=1000):
    # Inicializar métricas
    victorias_dqn = 0
    victorias_ql = 0
    empates = 0
    total_recompensa_dqn = 0
    total_turnos = 0

    # Configuración del agente DQN
    state_dim = 20  # Asegúrate de que coincide con tu implementación
    action_dim = 5  # Número de acciones posibles
    agent_dqn = DeepQLearningAgent(state_dim, action_dim)
    agent_dqn.load_model(path)

    # Inicializar agente Q-Learning
    inicializar_Q()

    for episodio in range(episodios):
        # Inicializar partida
        cartas = cargar_cartas()
        cartas_dqn, cartas_ql = asignar_cartas(cartas)
        mano_dqn, mazo_dqn = copia.seleccionar_cartas_mano(cartas_dqn)
        mano_ql, mazo_ql = copia.seleccionar_cartas_mano(cartas_ql)

        victorias = {"User": {"Fuego": [], "Agua": [], "Nieve": []}, 
                     "IA": {"Fuego": [], "Agua": [], "Nieve": []}}
        historial_acciones = []
        done = False
        recompensa_dqn = 0
        turnos = 0

        while not done:
            # Obtener el estado actual
            estado_actual = get_state(victorias, mano_ql, mazo_dqn, mazo_ql, historial_acciones)
            estado_actual_encoded = encode_state(estado_actual)  # Codificar el estado para DQN

            # Selección de acciones
            accion_dqn = agent_dqn.select_action(estado_actual_encoded)
            accion_ql = copia.select_action(estado_actual, mano_ql)

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
                if resultado == "User":
                    recompensa_dqn += recompensa

            turnos += 1

            # Verificar si hay un ganador
            ganador, victoria = copia.verificar_condicion_victoria(victorias)
            if ganador:
                done = True
                if ganador == "User":
                    victorias_dqn += 1
                elif ganador == "IA":
                    victorias_ql += 1
                else:
                    empates += 1

            # Reemplazar cartas si quedan en el mazo
            if mazo_dqn and mazo_ql:
                nueva_carta_dqn = np.random.choice(mazo_dqn)
                mazo_dqn.remove(nueva_carta_dqn)
                mano_dqn.append(nueva_carta_dqn)

                nueva_carta_ql = np.random.choice(mazo_ql)
                mazo_ql.remove(nueva_carta_ql)
                mano_ql.append(nueva_carta_ql)

        total_recompensa_dqn += recompensa_dqn
        total_turnos += turnos

    # Calcular métricas finales
    win_rate_dqn = victorias_dqn / episodios * 100
    win_rate_ql = victorias_ql / episodios * 100
    empate_rate = empates / episodios * 100
    promedio_recompensa_dqn = total_recompensa_dqn / episodios
    promedio_turnos = total_turnos / episodios

    print(f"\nResultados del modelo {modelo_dqn}:")
    print(f"Victorias DQN: {victorias_dqn} ({win_rate_dqn:.2f}%)")
    print(f"Victorias Q-Learning: {victorias_ql} ({win_rate_ql:.2f}%)")
    print(f"Empates: {empates} ({empate_rate:.2f}%)")
    print(f"Recompensa promedio DQN: {promedio_recompensa_dqn:.2f}")
    print(f"Turnos promedio por juego: {promedio_turnos:.2f}")

# Evaluar modelos DQN entrenados contra Q-Learning
evaluar_modelo_dqn_vs_qlearning('dqn_trained_model_1000','models_QL/dqn_trained_model_1000.pth' ,episodios=5000)
evaluar_modelo_dqn_vs_qlearning('dqn_trained_model_5000','models_QL/dqn_trained_model.pth', episodios=5000)
evaluar_modelo_dqn_vs_qlearning('dqn_trained_model_10000','models_QL/dqn_trained_model_10000.pth', episodios=5000)

# Evaluar modelos DQN con Self-Play contra Q-Learning
evaluar_modelo_dqn_vs_qlearning('dqn_trained_selfplay_model_6000', 'models_QL_Self/dqn_trained_selfplay_model_1000.pth',episodios=5000)
evaluar_modelo_dqn_vs_qlearning('dqn_trained_selfplay_model_15000','models_QL_Self/dqn_trained_selfplay_model.pth',episodios=5000)
evaluar_modelo_dqn_vs_qlearning('dqn_trained_selfplay_model_30000', 'models_QL_Self/dqn_trained_selfplay_model_10000.pth',episodios=5000)

# Evaluar modelos DQN con Self-Play desde cero contra Q-Learning
evaluar_modelo_dqn_vs_qlearning('dqn_trained_selfplay_model_cero_5000', 'models_self/dqn_trained_selfplay_cero_5000.pth',episodios=5000)
evaluar_modelo_dqn_vs_qlearning('dqn_trained_selfplay_model_cero_10000','models_self/dqn_trained_selfplay_cero_10000.pth',episodios=5000)
evaluar_modelo_dqn_vs_qlearning('dqn_trained_selfplay_model_cero_15000', 'models_self/dqn_trained_selfplay_cero_15000.pth',episodios=5000)
