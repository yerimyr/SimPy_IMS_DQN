import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from GymWrapper import GymInterface  
import gym
import environment as env
from config_RL import *
from config_SimPy import *

# Hyperparameters
gamma = 0.95  # Discount factor
batch_size = 20  # Minibatch size
learning_rate = 0.0001  # Learning rate
buffer_capacity = 1000000  # Replay buffer capacity
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.1  # Final exploration rate
epsilon_decay = 0.998  # Exploration rate decay
target_update_interval = 7000  # Target network update interval

# Environment
env = GymInterface()  # GymInterface 클래스를 인스턴스화하여 환경 객체 env를 생성함함
state_dim = env.observation_space.shape[0]  # observation_space의 차원인 상태 벡터의 크기를 추출하여 state_dim에 저장함
action_dim = env.action_space.n  # action_space에서 가능한 행동의 개수를 추출하여 action_dim에 저장함.

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 고정된 크기의 deque를 생성하고 최대 크기는 capacity크리고 지정, maxlen을 초과하면 가장 오래된 요소가 자동으로 제거됨.
    
    def push(self, state, action, reward, next_state, done):  # buffer에 새로운 경험을 저장하는 push 메서드
        self.buffer.append((state, action, reward, next_state, done))  # buffer에 (state, action, reward, next_state, done)인 튜플 형식으로 경험을 저장함.
    
    def sample(self, batch_size):  # buffer에서 랜덤하게 batch_size개의 경험을 샘플링하는 sample 메서드.
        batch = random.sample(self.buffer, batch_size)  # buffer에서 중복 없이 랜덥하게 batch_size개의 경험을 선택함.
        states, actions, rewards, next_states, dones = zip(*batch)  # 샘플링된 경험 리스트를 states, actions, rewards, next_states, dones로 분리함.
        return (  # 각각의 데이터를 numpy 배열로 변환하여 배치 처리가 가능하도록 함.
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):  # buffer에 저장된 경험의 개수를 반환하는 __len__메서드
        return len(self.buffer)  # deque객체인 buffer의 길이를 반환

# Q-Network
class QNetwork(nn.Module):  # pytorch의 신경망 모듈인 nn.Module을 상속받아 정의된 클래스임.
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()  # 부모 클래스인 nn.Module의 초기화 메서드를 호출함.
        self.fc1 = nn.Linear(state_dim, 128)  # 입력층: state_dim을 입력 받아 첫 번째 은닉층으로 전달함. 이 때 출력 크기는 128.
        self.fc2 = nn.Linear(128, 128)  # 은닉층: 첫 번째 은닉층의 출력을 처리하여 다시 128개의 출력값을 출력함.
        self.fc3 = nn.Linear(128, action_dim)  # 출력층: 128개의 입력값을 받아서 다시 각 행동에 대한 q값을 출력함.
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 입력층: 입력 상태 x를 fc1에 전달함(활성화 함수로 relu를 적용함)
        x = torch.relu(self.fc2(x))  # 은닉층: fc1의 출력을 fc2에 전달함(활성화 함수로 relu를 적용함)
        return self.fc3(x)  # 출력층: fc2의 출력을 fc3에 전달함(활성화 함수가 없는 선형 출력 -> 각 행동에 대한 q값을 계산하는 거지 q값의 범위를 제한할 필요가 없기 때문에 활성화 함수를 사용하지 않음)
 
# Initialize Networks and Optimizer
q_net = QNetwork(state_dim, action_dim)  # q_net을 초기화하여 학습할 신경망을 생성함.
target_net = QNetwork(state_dim, action_dim)  # 타겟 네트워크를 초기화.
target_net.load_state_dict(q_net.state_dict())  # q_net의 가중치를 load_state_dict를 사용하여 타겟 네트워크로 복사함.
optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)  # q_net을 학습하기 위한 옵티마이저 설정.(optim.Adam을 사용하여 학습 속도와 가중치 업데이트를 효율적으로 조정함함)

# Replay Buffer Initialization
buffer = ReplayBuffer(buffer_capacity)  # 경험 리플레이 버퍼를 생성하여 경험을 저장함.

# Exploration rate initialization
epsilon = epsilon_start  # 탐험률의 초기값 설정.

# Training Loop
for episode in range(N_EPISODES):  # 각 에피소드 동안 에이전트는 환경을 초기화하고, state를 갱신하며 학습함.
    state = env.reset()  # 환경을 초기화하고, 초기 상태를 반환함.
    total_reward = 0  # 현재 에피소드 동안 받은 총 보상을 누적함.
    for t in range(SIM_TIME):  # 에피소드에서 최대 실행 가능한 타입스텝의 수를 받아, 각 타임스텝에서 행동을 선택하고 환경과 상호작용하며 학습함.
        # ε-greedy action selection
        if random.random() < epsilon:  # 탐험
            action = env.action_space.sample()  # Random action
        else:  # 활용
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # pytorch 모델 입력은 배치 형식이 필요하므로 state를 2d 텐서로 변환 
            with torch.no_grad():  # q값 계산 시 메모리 절약을 위해 학습을 위한 그래프프를 생성하지 않도록 설정.
                action = q_net(state_tensor).argmax(dim=1).item()  # Greedy action

        # Take action in environment
        next_state, reward, done, _ = env.step(action)  # 선택된 action을 env에 전달하여 next_state(학습 루프에서는 next_state의 형태로 buffer에 저장하여 샘플링되어 이용되고 추후 타겟 q값을 계산할 때 next_state가 필요하므로 next_state 형태로 반환되는 것), reward, done을 반환함.
        total_reward += reward  # 현재 에피소드의 총 보상을 업데이트함.

        # Store transition in replay buffer
        buffer.push(state, action, reward, next_state, done)  # state, action, reward, next_state, done을 replay buffer에 저장함.
        state = next_state  # next_state를 현재 state로 업데이트함.

        # Update Q-Network
        if len(buffer) >= batch_size:  # buffer에 저장된 경험이 batch_size 이상일 때만 학습을 진행함.
            # Sample minibatch from replay buffer
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)  # buffer에서 랜덤하게 batch_size만큼 샘플링
            states = torch.FloatTensor(states)  # numpy 배열로 반환된 데이터를 pytorch 텐서로 변환함.
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Compute target Q-values
            with torch.no_grad():
                next_q_values = target_net(next_states).max(dim=1)[0]  # 타겟 네트워크를 통해 다음 상태의 q값 계산.(max(dim=1)[0]을 사용하여 가능한 모든 행동 중 가장 큰 q값 선택)
                target_q_values = rewards + gamma * next_q_values * (1 - dones)  # 벨만 방정식을 사용하여 타겟 q값 계산

            # Compute current Q-values
            current_q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # q_net을 사용하여 현재 states의 모든 q값을 계산하고, gather을 사용하여 샘플링된 actions에 해당하는 q값을 선택함.

            # Compute loss
            loss = nn.MSELoss()(current_q_values, target_q_values)  # 손실함수를 통해 손실 계산

            # Optimize the Q-network
            optimizer.zero_grad()  # 이전 단계에서 계산된 gradient를 초기화함.
            loss.backward()  # 손실의 gradient 계산.
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)  # Gradient clipping
            optimizer.step()  # 계산된 gradient를 통해 네트워크 가중치 업데이트.

        # Update target network
        if t % target_update_interval == 0:  # target_update_interval마다 타겟 네트워크를 q_net으로 업데이트
            target_net.load_state_dict(q_net.state_dict())

        if done:  # 환경에서 에피소드가 종료되면 타임스텝 루프를 빠져나감.
            break

    # Decay exploration rate
    epsilon = max(epsilon_end, epsilon * epsilon_decay)  # 학습이 진행됨에 따라 탐험률을 점진적으로 감소함.
  
    print(f"Episode {episode}, Total Reward: {total_reward}")

# Evaluation -> 학습된 q_net을 사용해 환경에서 에이전트의 성능을 평가함.
def evaluate(env, model, N_EVAL_EPISODES=100):  
    total_rewards = []  # 각 에피소드에서 받은 총 보상을 저장하기 위한 리스트.
    for _ in range(N_EVAL_EPISODES):  # 설정된 episodes 수만큼 평가를 반복함.
        state = env.reset()  # 환경 초기화 및 초기 상태 반환.
        done = False  # 에피소드 종료 여부 초기화.
        total_reward = 0  # 현재 에피소드의 총 보상을 누적하기 위한 변수 초기화.
        while not done:  # 에피소드가 종료될 때까지 반복하여 실행.
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # state를 네트워크 입력인 pytorch 텐서로 변환함.
            with torch.no_grad():  # evaluate 중에는 메모리와 속도 최적화를 위해 gradient를 계산하기 않도록 설정.
                action = model(state_tensor).argmax(dim=1).item()  # q_net을 통해 state에 대한 q값 예측, 가장 큰 q값을 가지는 action 선택.(탐험 없이 순수 활용), .item을 통해 텐서에서 값을 추출하여 python 정수로 변환함.
            state, reward, done, _ = env.step(action)  # 선택한 action을 env에 전달하여 state(evaluate루프기 때문에 next_state가 아닌 state), reward, done을 반환함.
            total_reward += reward  # 현재 에피소드의 총 보상을 업데이트.
        total_rewards.append(total_reward)  # 현재 에피소드의 총 보상을 total_rewards 리스트에 저장.
    print(f"Average Reward: {np.mean(total_rewards)}")  # 모든 에피소드에서의 총 보상의 평균값 계산.

# Evaluate the trained model 
evaluate(env, q_net)  # 평가 함수 호출.
