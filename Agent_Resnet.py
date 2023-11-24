import random 
import torch.optim as optim
import torch.nn.functional as F
from collections import deque 
from QNetwork import QNetwork
import torch
import numpy as np
from torchvision import models
import torch.nn as nn


num_classes = 5
# Load pre-trained ResNet without the top classification layer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(pretrained=True).to(device)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes).to(device)  # Replace the top layer
class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze the weights of the ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Modify the classifier layers
        self.fc1 = nn.Linear(self.resnet.fc.out_features, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        #x = torch.permute(x, (0, 3, 1, 2))
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class Agent():
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnetwork_local = CustomResnet().to(device)
        self.qnetwork_target = CustomResnet().to(device)
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
        state = torch.permute(state, (0, 3, 1, 2))
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

        # Specifically in the case of the Q-Network the predictions will be the Q-Values of taking each actions
        # Here in our case actions contains the indices of the taken action.
        # The first argument is the dimension along which the gathering is performed. 1, i.e. columns in this case.
        # The second argument, actions, contains the indices that determine which elements to select from the predictions tensor.
        predictions = self.qnetwork_local(torch.permute(torch.from_numpy(states), (0, 3, 1, 2)).float().to(device)).gather(1,actions)

        # After we take highest Qs if we pass the next_states through the network
        with torch.no_grad():
            # This is equivalent to $max_{a}Q(s_{t+1}, a) - Q(s_t, a_t)$
            q_next = self.qnetwork_target(torch.permute(torch.from_numpy(next_states), (0, 3, 1, 2)).float().to(device)).detach().max(1)[0].unsqueeze(1)
        
        targets = rewards + (self.gamma * q_next * (1-dones))
        targets = targets.float()
        loss = criterion(predictions,targets).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


