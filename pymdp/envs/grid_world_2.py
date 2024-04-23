import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import seaborn as sns

from pymdp.envs import Env
from pymdp import utils, maths


LOCATION_FACTOR_ID = 0
TRIAL_FACTOR_ID = 1

LOCATION_MODALITY_ID = 0
REWARD_MODALITY_ID = 1

REWARD_IDX = 1
LOSS_IDX = 2

# ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

class FrozenLake_Custom(Env):
    """ States:
            Location - All the individual locations of the grid ( Eg. 1x9 vector for 3x3 grid)
            Context - To specify the hole and goal interchanging (1x2 vector)
        
        Obs:
            Location - Identical to location hidden state (Eg. 9 categorical location one-hots for 3x3 grid)
            Reward - [No_reward, Reward, Loss]
    """
    
    def __init__(self, reward_condition=1):
        self.grid_dims = [3, 3]
        self.num_locations = np.prod(self.grid_dims)
        self.num_reward_conditions = 2
        self.num_states = [self.num_locations, self.num_reward_conditions]
        self.num_controls = [len(ACTIONS), 1]
        self.num_obs = [self.num_locations, self.num_reward_conditions + 1]
        self.num_factors = len(self.num_states)
        self.num_modalities = len(self.num_obs)


        # create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates 
        grid = np.arange(self.num_grid_points).reshape(self.grid_dims)
        it = np.nditer(grid, flags=["multi_index"])

        self.loc_list = []
        while not it.finished:
            self.loc_list.append(it.multi_index)
            it.iternext()

        (self.reward_loc, self.hole_loc) = self.set_reward_locs(reward_condition)
        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

        self._reward_condition = None
        self._state = None
    
    def reset(self, state=None, reward_condition=None):
        if state is None:
            loc_state = utils.onehot(0, self.num_locations)
            
            self._reward_condition = np.random.randint(self.num_reward_conditions) if reward_condition is None \
                                     else reward_condition
            reward_condition_categorical = utils.onehot(self._reward_condition, self.num_reward_conditions)
            # TODO: Should the A and B matrices change??
            
            full_state = utils.obj_array(self.num_factors)
            full_state[LOCATION_FACTOR_ID] = loc_state
            full_state[TRIAL_FACTOR_ID] = reward_condition_categorical
            self._state = full_state
        else:
            self._state = state
        return self._get_observation()
    
    def reset_env(self, reward_condition):
        return self.reset(state=None,reward_condition=reward_condition)
    
    def set_reward_locs(self,reward_condition=1):
        (reward_loc, hole_loc) = (8, 6) if reward_condition==1 else (6, 8)
        return (reward_loc, hole_loc)

    def step(self, actions):
        prob_states = utils.obj_array(self.num_factors)
        for factor, state in enumerate(self._state):
            prob_states[factor] = self._transition_dist[factor][:, :, int(actions[factor])].dot(state)
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        return self._get_observation()

    def render(self):
        pass

    def sample_action(self):
        return [np.random.randint(self.num_controls[i]) for i in range(self.num_factors)]

    def get_likelihood_dist(self):
        return self._likelihood_dist

    def get_transition_dist(self):
        return self._transition_dist


    def get_rand_likelihood_dist(self):
        pass

    def get_rand_transition_dist(self):
        pass

    def _get_observation(self):

        prob_obs = [maths.spm_dot(A_m, self._state) for A_m in self._likelihood_dist]

        obs = [utils.sample(po_i) for po_i in prob_obs]
        return obs

    def _construct_transition_dist(self):

        # initialize the shapes of each sub-array `B[f]`
        B_f_shapes = [ [ns, ns, self.num_controls[f]] for f, ns in enumerate(self.num_states)]

        # create the `B` array and fill it out
        B = utils.obj_array_zeros(B_f_shapes)

        # fill out `B[0]` using the 
        for action_id, action_label in enumerate(ACTIONS):

            for curr_state, grid_location in enumerate(self.loc_list):

                y, x = grid_location

                if action_label == "UP":
                    next_y = y - 1 if y > 0 else y 
                    next_x = x
                elif action_label == "DOWN":
                    next_y = y + 1 if y < (self.grid_dims[0]-1) else y 
                    next_x = x
                elif action_label == "LEFT":
                    next_x = x - 1 if x > 0 else x 
                    next_y = y
                elif action_label == "RIGHT":
                    next_x = x + 1 if x < (self.grid_dims[1]-1) else x 
                    next_y = y
                elif action_label == "STAY":
                    next_x = x
                    next_y = y

                new_location = (next_y, next_x)
                next_state = self.loc_list.index(new_location)
                B[LOCATION_FACTOR_ID][next_state, curr_state, action_id] = 1.0
        
        B[TRIAL_FACTOR_ID][:,:,0] = np.eye(self.num_states[1])

        return B

    def _construct_likelihood_dist(self):

        A = utils.obj_array_zeros([[obs_dim] + self.num_states for obs_dim in self.num_obs])
        
        # make the location observation only depend on the location state (proprioceptive observation modality)
        A[LOCATION_MODALITY_ID] = np.tile(np.expand_dims(np.eye(self.num_locations), (-1)), (1, 1, self.num_states[1]))

        ## Reward addition
        # make the reward observation depend on the location (being at reward location) and the reward condition
        A[REWARD_MODALITY_ID][0,:,:] = 1.0  # default makes Null the most likely observation everywhere.
                                            # 1 for almost all the locations with 'No_reward' obs
        
        reward_loc_idx = self.loc_list.index(self.reward_loc)
        hole_loc_idx = self.loc_list.index(self.hole_loc)

        # Setting appropriate values for the reward location
        A[REWARD_MODALITY_ID][0,reward_loc_idx,:] = 0.0
        A[REWARD_MODALITY_ID][1,reward_loc_idx,0] = 1.0
        A[REWARD_MODALITY_ID][2,reward_loc_idx,1] = 1.0

        # Setting appropriate values for the hole location
        A[REWARD_MODALITY_ID][0,hole_loc_idx,:] = 0.0
        A[REWARD_MODALITY_ID][1,hole_loc_idx,1] = 1.0
        A[REWARD_MODALITY_ID][2,hole_loc_idx,0] = 1.0

        return A

    def _construct_state(self, state_tuple):

        state = utils.obj_array(self.num_factors)
        for f, ns in enumerate(self.num_states):
            state[f] = utils.onehot(state_tuple[f], ns)

        return state

    @property
    def state(self):
        return self._state

    @property
    def reward_condition(self):
        return self._reward_condition

class GridWorldCueEnv():    
    
    def __init__(self, grid_dims = [5, 7], 
                 cue1_loc = (2, 0), cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)],
                 reward_locations = [(1, 5), (3, 5)],
                 starting_loc = (0,0), cue2 = 'L1', reward_condition = 'A'):

        self.init_loc = starting_loc
        self.current_location = self.init_loc
        self.grid_dims = grid_dims # dimensions of the grid (number of rows, number of columns)
        self.num_grid_points = np.prod(grid_dims) # total number of grid locations (rows X columns)
        # create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates 
        grid = np.arange(self.num_grid_points).reshape(grid_dims)
        it = np.nditer(grid, flags=["multi_index"])

        self.loc_list = []
        while not it.finished:
            self.loc_list.append(it.multi_index)
            it.iternext()

        self.cue1_loc = cue1_loc
        self.cue2_name = cue2
        self.cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
        self.cue2_locations = cue2_locations
        self.cue2_loc = cue2_locations[self.cue2_loc_names.index(self.cue2_name)]

        # names of the reward conditions and their locations
        self.reward_conditions = ["A", "B"]
        self.reward_locations = reward_locations
        self.reward_condition = reward_condition
        print(f'Starting location is {self.init_loc}, Reward condition is {self.reward_condition}, cue is located in {self.cue2_name}')

        # list of dimensionalities of the hidden states -- useful for creating generative model later on
        self.num_states = [self.num_grid_points, len(cue2_locations), len(self.reward_conditions)]

        # Names of the cue1 observation levels, the cue2 observation levels, and the reward observation levels
        self.cue1_names = ['Null'] + self.cue2_loc_names # signals for the possible Cue 2 locations, that only are seen when agent is visiting Cue 1
        self.cue2_names = ['Null', 'reward_in_A', 'reward_in_B']
        self.reward_names = ['Null', 'Cheese', 'Shock']
        self.num_obs = [self.num_grid_points, len(self.cue1_names), len(self.cue2_names), len(self.reward_names)]
    
    def step(self,action_label):

        (Y, X) = self.current_location

        if action_label == "UP": 
            Y_new = Y - 1 if Y > 0 else Y
            X_new = X

        elif action_label == "DOWN": 

            Y_new = Y + 1 if Y < (self.grid_dims[0]-1) else Y
            X_new = X

        elif action_label == "LEFT": 
            Y_new = Y
            X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
            Y_new = Y
            X_new = X +1 if X < (self.grid_dims[1]-1) else X

        elif action_label == "STAY":
            Y_new, X_new = Y, X 
        
        self.current_location = (Y_new, X_new) # store the new grid location

        loc_obs = self.current_location # agent always directly observes the grid location they're in 

        if self.current_location == self.cue1_loc:
            cue1_obs = self.cue2_name
        else:
            cue1_obs = 'Null'

        if self.current_location == self.cue2_loc:
            cue2_obs = self.cue2_names[self.reward_conditions.index(self.reward_condition)+1]
        else:
            cue2_obs = 'Null'
        
        # @NOTE: here we use the same variable `reward_locations` to create both the agent's generative model (the `A` matrix) as well as the generative process. 
        # This is just for simplicity, but it's not necessary -  you could have the agent believe that the Cheese/Shock are actually stored in arbitrary, incorrect locations.

        if self.current_location == self.reward_locations[0]:
            if self.reward_condition == 'A':
                reward_obs = 'Cheese'
            else:
                reward_obs = 'Shock'
        elif self.current_location == self.reward_locations[1]:
            if self.reward_condition == 'B':
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
    
    def plot_empty_grid(self):
        fig, ax = plt.subplots(figsize=(10, 6)) 

        # create the grid visualization
        X, Y = np.meshgrid(np.arange(self.grid_dims[1]+1), np.arange(self.grid_dims[0]+1))
        h = ax.pcolormesh(X, Y, np.ones(self.grid_dims), edgecolors='k', vmin = 0, vmax = 30, linewidth=3, cmap = 'coolwarm')
        ax.invert_yaxis()

        # Put gray boxes around the possible reward locations
        reward_A = ax.add_patch(patches.Rectangle((self.reward_locations[0][1],self.reward_locations[0][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor=[0.5, 0.5, 0.5]))
        reward_B = ax.add_patch(patches.Rectangle((self.reward_locations[1][1],self.reward_locations[1][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor=[0.5, 0.5, 0.5]))
        reward_condition = self.reward_condition
        ax.text(self.reward_locations[0][1] + 0.5, self.reward_locations[0][0] + 0.5, 'A', ha='center', va='center', fontsize = 15)
        ax.text(self.reward_locations[1][1] + 0.5, self.reward_locations[1][0] + 0.5, 'B', ha='center', va='center', fontsize = 15)

        text_offsets = [0.4, 0.6]

        cue_grid = np.ones(self.grid_dims)
        cue_grid[self.cue1_loc[0],self.cue1_loc[1]] = 15.0
        for ii, loc_ii in enumerate(self.cue2_locations):
            row_coord, column_coord = loc_ii
            cue_grid[row_coord, column_coord] = 5.0
            ax.text(column_coord+text_offsets[0], row_coord+text_offsets[1], self.cue2_loc_names[ii], fontsize = 15, color='k')
        h.set_array(cue_grid.ravel())
    
    def plot_movement(self, history_of_locs, T=10):
        all_locations = np.vstack(history_of_locs).astype(float) # create a matrix containing the agent's Y/X locations over time (each coordinate in one row of the matrix)

        fig, ax = plt.subplots(figsize=(10, 6)) 

        # create the grid visualization
        X, Y = np.meshgrid(np.arange(self.grid_dims[1]+1), np.arange(self.grid_dims[0]+1))
        h = ax.pcolormesh(X, Y, np.ones(self.grid_dims), edgecolors='k', vmin = 0, vmax = 30, linewidth=3, cmap = 'coolwarm')
        ax.invert_yaxis()

        # get generative process global parameters (the locations of the Cues, the reward condition, etc.)
        cue1_loc, cue2_loc, reward_condition = self.cue1_loc, self.cue2_loc, self.reward_condition
        reward_A = ax.add_patch(patches.Rectangle((self.reward_locations[0][1],self.reward_locations[0][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))
        reward_B = ax.add_patch(patches.Rectangle((self.reward_locations[1][1],self.reward_locations[1][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor='none'))
        reward_loc = self.reward_locations[0] if reward_condition == "A" else self.reward_locations[1]

        if reward_condition == "A":
            reward_A.set_edgecolor('g')
            reward_A.set_facecolor('g')
            reward_B.set_edgecolor([0.7, 0.2, 0.2])
            reward_B.set_facecolor([0.7, 0.2, 0.2])
        elif reward_condition == "B":
            reward_B.set_edgecolor('g')
            reward_B.set_facecolor('g')
            reward_A.set_edgecolor([0.7, 0.2, 0.2])
            reward_A.set_facecolor([0.7, 0.2, 0.2])
        reward_A.set_zorder(1)
        reward_B.set_zorder(1)

        text_offsets = [0.4, 0.6]
        cue_grid = np.ones(self.grid_dims)
        cue_grid[cue1_loc[0],cue1_loc[1]] = 15.0
        for ii, loc_ii in enumerate(self.cue2_locations):
            row_coord, column_coord = loc_ii
            cue_grid[row_coord, column_coord] = 5.0
            ax.text(column_coord+text_offsets[0], row_coord+text_offsets[1], self.cue2_loc_names[ii], fontsize = 15, color='k')
        
        h.set_array(cue_grid.ravel())

        cue1_rect = ax.add_patch(patches.Rectangle((cue1_loc[1],cue1_loc[0]),1.0,1.0,linewidth=8,edgecolor=[0.5, 0.2, 0.7],facecolor='none'))
        cue2_rect = ax.add_patch(patches.Rectangle((cue2_loc[1],cue2_loc[0]),1.0,1.0,linewidth=8,edgecolor=[0.5, 0.2, 0.7],facecolor='none'))

        ax.plot(all_locations[:,1]+0.5,all_locations[:,0]+0.5, 'r', zorder = 2)

        temporal_colormap = cm.hot(np.linspace(0,1,T+1))
        dots = ax.scatter(all_locations[:,1]+0.5,all_locations[:,0]+0.5, 450, c = temporal_colormap, zorder=3)

        ax.set_title(f"Cue 1 located at {cue2_loc}, Cue 2 located at {cue2_loc}, Cheese on {reward_condition}", fontsize=16)
