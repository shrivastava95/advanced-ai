import common.game_constants as game_constants
import common.game_state as game_state
# import game_constants
# import game_state
import pygame

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device, "\n")

class KeyboardController:
    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        keys = pygame.key.get_pressed()
        action = game_state.GameActions.No_action
        if keys[pygame.K_LEFT]:
            action = game_state.GameActions.Left
        if keys[pygame.K_RIGHT]:
            action = game_state.GameActions.Right
        if keys[pygame.K_UP]:
            action = game_state.GameActions.Up
        if keys[pygame.K_DOWN]:
            action = game_state.GameActions.Down
    
        return action


class AIController:
### ------- You can make changes to this file from below this line --------------

    def __init__(self) -> None:
        # Add more lines to the constructor if you need
        self.state_size = 6 + 4*game_constants.ENEMY_COUNT
        self.num_actions = 5
        self.qnetwork = QNetwork(self.state_size, self.num_actions)
        self.qnetwork = self.qnetwork.to(device)

        self.discount = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 1e-5

        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criteria = nn.SmoothL1Loss()
        self.criteria = self.criteria.to(device)
        self.replay_memory = deque(maxlen=10000)
    

    def GetGoodState(self, state:game_state.GameState):
        st = [
            # goal location
            state.GoalLocation.x,
            state.GoalLocation.y,
            # player location
            state.PlayerEntity.entity.x,
            state.PlayerEntity.entity.y,
            # player velocity
            state.PlayerEntity.velocity.x,
            state.PlayerEntity.velocity.y,
        ]
        for enemy in state.EnemyCollection:
            # enemy location
            st.append(enemy.entity.x)
            st.append(enemy.entity.y)
            # enemy velocity
            st.append(enemy.velocity.x)
            st.append(enemy.velocity.y)

        return st


    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        # This function should select the best action at a given state
        if np.random.uniform() < self.epsilon:
            act = np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                state = self.GetGoodState(state)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state_tensor = state_tensor.to(device)
                q_values = self.qnetwork(state_tensor)
                act =  q_values.argmax().item()

        acts = {
            0: game_state.GameActions.No_action,
            1: game_state.GameActions.Left,
            2: game_state.GameActions.Right,
            3: game_state.GameActions.Up,
            4: game_state.GameActions.Down,
        }
        # A wrong example (just so you can compile and check)
        # return game_state.GameActions.Right
        return acts[act]
    

    def remember(self, state, action, reward, new_state, done):
        self.replay_memory.append((state, action, reward, new_state, done))


    def train(self,batch_size):
        if len(self.replay_memory) < batch_size:
            batch = self.replay_memory
        else:
            batch = random.sample(self.replay_memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        q_values = self.qnetwork(states)
        with torch.no_grad():
            next_q_values = self.qnetwork(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.discount * next_q_value * (1 - dones)

        loss = self.criteria(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        return loss.item()
    
    def _euclide(self,x1,x2,y1,y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def GetReward(self, new_state, state, obs):
        # Complete this function
        reward = 0
        if obs == game_state.GameObservation.Reached_Goal:
            reward = 100
        elif obs == game_state.GameObservation.Enemy_Attacked:
            reward = -1000
        elif obs == game_state.GameObservation.Nothing:
            reward = -5
            # reward = 0

            # if player is closer to goal
            old_dist = self._euclide(
                state.PlayerEntity.entity.x,state.GoalLocation.x,
                state.PlayerEntity.entity.y, state.GoalLocation.y
            )
            new_dist = self._euclide(
                new_state.PlayerEntity.entity.x, new_state.GoalLocation.x,
                new_state.PlayerEntity.entity.y, new_state.GoalLocation.y
            )
            # if new_dist < old_dist:
            reward += 5 * ( 1 - new_dist / max(game_constants.GAME_HEIGHT, game_constants.GAME_WIDTH)) ** 2
            # reward +=  (old_dist - new_dist) / 1000

            
            # # # if player is moving towards goal
            # # Vx = state.PlayerEntity.velocity.x,
            # # Vy = state.PlayerEntity.velocity.y,

            # # Vx, Vy = Vx[0], Vy[0]
            # # V = (Vx**2 + Vy**2)**0.5
            # # if V != 0:
            # #     Vx, Vy = Vx / V, Vy / V
            # #     Vangle = np.angle(complex(Vx, Vy))

            # #     Sx = state.GoalLocation.x - state.PlayerEntity.entity.x
            # #     Sy = state.GoalLocation.y - state.PlayerEntity.entity.y
            # #     S = (Sx**2 + Sy**2)**0.5
            # #     if S != 0:
            # #         Sx, Sy = Sx / S, Sy / S
            # #         Sangle = np.angle(complex(Sx, Sy))

            # #         reward += ( 1 - abs(Vangle - Sangle) ) * 10


            
            # # if player is closer to enemy
            # safe_dist = (game_constants.ENEMY_SIZE + game_constants.PLAYER_SIZE) * 4
            # for enemy in new_state.EnemyCollection:
            #     dist = self._euclide(
            #         enemy.entity.x, new_state.PlayerEntity.entity.x,
            #         enemy.entity.y, new_state.PlayerEntity.entity.y
            #     )
            #     if dist < safe_dist:
            #         # reward = -10  # might be too high. agent tends to stick to corners and not pursue goal
            #         reward += -10 * (1 - dist / safe_dist)

        return reward


    def TrainModel(self):
        # Complete this function
        epochs = 100
        max_steps_per_epoch = 1000
        batch_size = 64

        for _ in range(epochs):

            state = game_state.GameState()
            done = False
            total_reward = 0
            avg_loss = 0

            steps = 0
            for __ in range(max_steps_per_epoch):
                action = self.GetAction(state)
                og_state = deepcopy(state)
                obs = state.Update(action)
                reward = self.GetReward(state,og_state,obs)
                # done = (obs == game_state.GameObservation.Enemy_Attacked or obs == game_state.GameObservation.Reached_Goal)
                done = False

                action_num = {
                    game_state.GameActions.No_action: 0,
                    game_state.GameActions.Left: 1,
                    game_state.GameActions.Right: 2,
                    game_state.GameActions.Up: 3,
                    game_state.GameActions.Down: 4,
                }[action]   

                self.remember(self.GetGoodState(og_state), action_num, reward, self.GetGoodState(state), done)
                avg_loss += self.train(batch_size)


                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            avg_loss /= steps

            if done:
                if obs == game_state.GameObservation.Enemy_Attacked:
                    cause = "Enemy_Attacked"
                else:
                    cause = "Reached_Goal"
            else:
                cause = "Max_Steps"
            # print("Epoch: [{}/{}]\t| Total Reward: {:.3f}\t| Average Loss: {:.3f}\t| Cause: {}\t| Steps: {}"\
            #       .format(_,epochs,total_reward,avg_loss,cause,steps))
            print("Epoch: [{}/{}]\t| Average Loss: {:.3f}\t| Cause: {}\t| Steps: {}"\
                    .format(_,epochs,avg_loss,cause,steps))


class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values
    
### ------- You can make changes to this file from above this line --------------

    # This is a custom Evaluation function. You should not change this function
    # You can add other methods, or other functions to perform evaluation for
    # yourself. However, this evalution function will be used to evaluate your model
    def EvaluateModel(self):
        attacked = 0
        reached_goal = 0
        state = game_state.GameState()
        for _ in range(100000):
            action = self.GetAction(state)
            obs = state.Update(action)
            if(obs==game_state.GameObservation.Enemy_Attacked):
                attacked += 1
            elif(obs==game_state.GameObservation.Reached_Goal):
                reached_goal += 1
        return (attacked, reached_goal)

