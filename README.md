# Car Racing Deep Q Network

This code implements an agent to solve the gym's CarRacing problem. Here we implemented 2 different networks architectures to compare their performance in the same agent.

## Deep Q Network Custom Architecture
The custom architecture is a CNN to extract features of the state, that is in this case the image of the racing car in the speedway. Further, the features are flattered and within some fully connected layers this network will yield a tensor of size five, with the Q-Value for each of the 5 possible actions.
The Architecture of the network can be seen below.
![Alt text](DQN.png)

<div align="center">  
  <img align="center" alt="GIF" src="https://github.com/rs-benatti/CarRacing_Deep_Q_Network/blob/master/Custom_QNet_Learner_Animation.gif"
 width="400" height="400" />
</div>

## Resnet Architecture
The other architectur used is a modified ResNet18 model used for feature extraction from racing car images in a speedway environment. It leverages a pre-trained ResNet18 as the base architecture, freezing its weights to retain its learned features.

The modification includes adjusting the classifier layers to suit the specific needs of the task. After passing the input image through the ResNet layers, the output is flattened and fed into two fully connected layers. The first FC layer reduces the number of features to 256 while the ReLU activation function introduces non-linearity. Finally, the second FC layer produces an output tensor of size 5, representing the Q-Values corresponding to each of the 5 possible actions in the racing scenario.

This architecture is tailored for feature extraction from racing car images by utilizing a pre-trained ResNet18 and adapting its classifier layers to output the Q-Values necessary for decision-making in the racing environment. The network architecture is designed to efficiently process images and provide actionable Q-Value predictions for optimal racing actions.

The results can be seen below:

<div align="center">  
  <img align="center" alt="GIF" src="https://github.com/rs-benatti/CarRacing_Deep_Q_Network/blob/master/Resnet_Learner_Animation.gif"
 width="400" height="400" />
</div>


