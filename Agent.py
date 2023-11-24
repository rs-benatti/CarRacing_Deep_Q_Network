import random 
import torch.optim as optim
import torch.nn.functional as F
from collections import deque 
from QNetwork import QNetwork
import torch
import numpy as np

class Agent():
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnetwork_local = QNetwork().to(device)
        self.qnetwork_target = QNetwork().to(device)
        self.memory = deque(maxlen=10000) 
        self.gamma = 0.97    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.batch_size = 512
        self.train_start = 3000
        
        self.counter_1 = 0
        self.counter_2 = 0

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        sample = random.random()
        if sample > self.epsilon:
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            action =  random.choice(np.arange(5))
            return action

    def step(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.counter_2 = (self.counter_2+1) % 500
        self.counter_1 = (self.counter_1+1) % 4

        if self.counter_2 == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        if self.counter_1 == 0 and len(self.memory) >= self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
            self.learn(minibatch)
    
    def learn(self, batch):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = torch.nn.MSELoss()

        states =  np.zeros((self.batch_size, 96, 96 ,3))
        next_states =  np.zeros((self.batch_size, 96, 96 ,3))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            state_i, action_i, reward_i, next_state_i, done_i = batch[i]
            states[i] = state_i
            next_states[i] = next_state_i  
            actions.append(action_i)
            rewards.append(reward_i)
            dones.append(done_i)
        

        actions = np.vstack(actions).astype(np.int)
        actions = torch.from_numpy(actions).to(device)

        rewards = np.vstack(rewards).astype(np.float)
        rewards = torch.from_numpy(rewards).to(device)

        dones = np.vstack(dones).astype(np.int)
        dones = torch.from_numpy(dones).to(device)



        self.qnetwork_local.train()
        self.qnetwork_target.eval()

        predictions = self.qnetwork_local(torch.from_numpy(states).float().to(device)).gather(1,actions)

        with torch.no_grad():
            q_next = self.qnetwork_target(torch.from_numpy(next_states).float().to(device)).detach().max(1)[0].unsqueeze(1)
        
        targets = rewards + (self.gamma * q_next * (1-dones))
        targets = targets.float()
        loss = criterion(predictions,targets).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


