import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Parámetros
GAMMA = 0.99   # Factor de descuento
LEARNING_RATE = 0.001  # Tasa de aprendizaje
EPSILON_START = 1.0  # Probabilidad inicial de exploración
EPSILON_END = 0.01  # Probabilidad mínima de exploración
EPSILON_DECAY = 0.995  # Factor de decaimiento de epsilon
BATCH_SIZE = 64  # Tamaño del mini-batch para entrenar
REPLAY_MEMORY_SIZE = 10000  # Tamaño del buffer de experiencia
TARGET_UPDATE = 10  # Frecuencia para actualizar la red de destino

# Red neuronal para aproximar la función Q
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2)
        return self.fc3(x)

class DeepQLearningAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Exploración: acción aleatoria
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()  # Explotación: acción basada en la red

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Tomar un mini-batch de experiencias de la memoria
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertir a tensores de PyTorch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Calcular Q(s, a)
        q_values = self.policy_net(states).gather(1, actions).squeeze()

        # Calcular Q-objetivo usando la red de destino
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Calcular la pérdida (loss)
        loss = F.mse_loss(q_values, target_q_values)

        # Actualizar la red neuronal
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reducir epsilon para menos exploración con el tiempo
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path='dqn_model.pth'):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path='dqn_model.pth'):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()
