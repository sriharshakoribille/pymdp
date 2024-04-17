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

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

class FrozenLake_Custom(Env):
    def __init__(self, context=1):
        self.grid_dims = [3, 3]
        self.num_locations = np.prod(self.grid_dims)
        self.num_states = [self.num_locations, 2]
        self.num_controls = [self.num_locations, 1]
        self.num_reward_conditions = self.num_states[TRIAL_FACTOR_ID]
        self.num_obs = [self.num_locations, self.num_reward_conditions + 1]
        self.num_factors = len(self.num_states)
        self.num_modalities = len(self.num_obs)

        (self.reward_loc, self.hole_loc) = (8, 6) if context==1 else (6, 8)

        # create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates 
        grid = np.arange(self.num_grid_points).reshape(self.grid_dims)
        it = np.nditer(grid, flags=["multi_index"])

        self.loc_list = []
        while not it.finished:
            self.loc_list.append(it.multi_index)
            it.iternext()

        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

        self._reward_condition = None
        self._state = None
    
    def reset(self, state=None):
        if state is None:
            loc_state = utils.onehot(0, self.num_locations)
            
            self._reward_condition = np.random.randint(self.num_reward_conditions) # randomly select a reward condition
            reward_condition = utils.onehot(self._reward_condition, self.num_reward_conditions)

            full_state = utils.obj_array(self.num_factors)
            full_state[LOCATION_FACTOR_ID] = loc_state
            full_state[TRIAL_FACTOR_ID] = reward_condition
            self._state = full_state
        else:
            self._state = state
        return self._get_observation()

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
        B_locs = np.eye(self.num_locations)
        B_locs = B_locs.reshape(self.num_locations, self.num_locations, 1)
        B_locs = np.tile(B_locs, (1, 1, self.num_locations))
        B_locs = B_locs.transpose(1, 2, 0)

        B = utils.obj_array(self.num_factors)

        B[LOCATION_FACTOR_ID] = B_locs
        B[TRIAL_FACTOR_ID] = np.eye(self.num_reward_conditions).reshape(
            self.num_reward_conditions, self.num_reward_conditions, 1
        )
        return B

    def _construct_likelihood_dist(self):

        A = utils.obj_array_zeros([[obs_dim] + self.num_states for obs_dim in self.num_obs])
        
        # make the location observation only depend on the location state (proprioceptive observation modality)
        A[LOCATION_MODALITY_ID] = np.tile(np.expand_dims(np.eye(self.num_locations), (-1)), (1, 1, self.num_states[1]))

        # make the reward observation depend on the location (being at reward location) and the reward condition
        A[REWARD_MODALITY_ID][0,:,:] = 1.0 # default makes Null the most likely observation everywhere


        # for loc in range(self.num_states[LOCATION_FACTOR_ID]):
        #     for reward_condition in range(self.num_states[TRIAL_FACTOR_ID]):

        #         # The case when the agent is in the centre location
        #         if loc == 0:
        #             # When in the centre location, reward observation is always 'no reward'
        #             # or the outcome with index 0
        #             A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

        #             # When in the centre location, cue is totally ambiguous with respect to the reward condition
        #             A[CUE_MODALITY_ID][:, loc, reward_condition] = 1.0 / self.num_obs[2]

        #         # The case when loc == 3, or the cue location ('bottom arm')
        #         elif loc == 3:

        #             # When in the cue location, reward observation is always 'no reward'
        #             # or the outcome with index 0
        #             A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

        #             # When in the cue location, the cue indicates the reward condition umambiguously
        #             # signals where the reward is located
        #             A[CUE_MODALITY_ID][reward_condition, loc, reward_condition] = 1.0

        #         # The case when the agent is in one of the (potentially-) rewarding armS
        #         else:

        #             # When location is consistent with reward condition
        #             if loc == (reward_condition + 1):
        #                 # Means highest probability is concentrated over reward outcome
        #                 high_prob_idx = REWARD_IDX
        #                 # Lower probability on loss outcome
        #                 low_prob_idx = LOSS_IDX
        #             else:
        #                 # Means highest probability is concentrated over loss outcome
        #                 high_prob_idx = LOSS_IDX
        #                 # Lower probability on reward outcome
        #                 low_prob_idx = REWARD_IDX

        #             reward_probs = self.reward_probs[0]
        #             A[REWARD_MODALITY_ID][high_prob_idx, loc, reward_condition] = reward_probs

        #             reward_probs = self.reward_probs[1]
        #             A[REWARD_MODALITY_ID][low_prob_idx, loc, reward_condition] = reward_probs

        #             # Cue is ambiguous when in the reward location
        #             A[CUE_MODALITY_ID][:, loc, reward_condition] = 1.0 / self.num_obs[2]

        #         # The agent always observes its location, regardless of the reward condition
        #         A[LOCATION_MODALITY_ID][loc, loc, reward_condition] = 1.0

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
