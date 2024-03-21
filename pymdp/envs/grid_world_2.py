import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pymdp.envs import Env

cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]
grid_dims = [5, 7]
cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']

# names of the reward conditions and their locations
reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(1, 5), (3, 5)]


class GridWorldCueEnv():    
    
    def __init__(self,starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L1', reward_condition = 'TOP'):

        self.init_loc = starting_loc
        self.current_location = self.init_loc

        self.cue1_loc = cue1_loc
        self.cue2_name = cue2
        self.cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
        self.cue2_loc = cue2_locations[self.cue2_loc_names.index(self.cue2_name)]

        self.reward_condition = reward_condition
        print(f'Starting location is {self.init_loc}, Reward condition is {self.reward_condition}, cue is located in {self.cue2_name}')
    
    def step(self,action_label):

        (Y, X) = self.current_location

        if action_label == "UP": 
          
          Y_new = Y - 1 if Y > 0 else Y
          X_new = X

        elif action_label == "DOWN": 

          Y_new = Y + 1 if Y < (grid_dims[0]-1) else Y
          X_new = X

        elif action_label == "LEFT": 
          Y_new = Y
          X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
          Y_new = Y
          X_new = X +1 if X < (grid_dims[1]-1) else X

        elif action_label == "STAY":
          Y_new, X_new = Y, X 
        
        self.current_location = (Y_new, X_new) # store the new grid location

        loc_obs = self.current_location # agent always directly observes the grid location they're in 

        if self.current_location == self.cue1_loc:
          cue1_obs = self.cue2_name
        else:
          cue1_obs = 'Null'

        if self.current_location == self.cue2_loc:
          cue2_obs = cue2_names[reward_conditions.index(self.reward_condition)+1]
        else:
          cue2_obs = 'Null'
        
        # @NOTE: here we use the same variable `reward_locations` to create both the agent's generative model (the `A` matrix) as well as the generative process. 
        # This is just for simplicity, but it's not necessary -  you could have the agent believe that the Cheese/Shock are actually stored in arbitrary, incorrect locations.

        if self.current_location == reward_locations[0]:
          if self.reward_condition == 'TOP':
            reward_obs = 'Cheese'
          else:
            reward_obs = 'Shock'
        elif self.current_location == reward_locations[1]:
          if self.reward_condition == 'BOTTOM':
            reward_obs = 'Cheese'
          else:
            reward_obs = 'Shock'
        else:
          reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs

    def reset(self):
        self.current_location = self.init_loc
        print(f'Re-initialized location to {self.init_loc}')
        loc_obs = self.current_location
        cue1_obs = 'Null'
        cue2_obs = 'Null'
        reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs