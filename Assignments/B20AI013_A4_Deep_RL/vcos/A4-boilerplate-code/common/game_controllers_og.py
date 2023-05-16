import common.game_constants as game_constants
import common.game_state as game_state
import pygame

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
        pass


    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        # This function should select the best action at a given state
        
        # A wrong example (just so you can compile and check)
        return game_state.GameActions.Right
    
    def TrainModel(self):
        # Complete this function
        epochs = 10 # You might want to change the number of epochs
        for _ in range(epochs):
            state = game_state.GameState()
            # Explore the state by updating it

            action = self.GetAction(state) # Select best action
            obs = state.Update(action) # obtain the observation made due to your action

            # You must complete this function by
            # training your model on the explored state,
            # using a suitable RL algorithm, and
            # by appropriately rewarding your model on 
            # the state that it lands

            pass
        pass


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