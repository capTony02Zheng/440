import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        #state = food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right
        if state == None or action == None:
            return
        ###safe check done
        food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = state
        
        
        self.N[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right,  action] += 1


    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        #state = food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right
        # similar for sprime
        if s == None or a == None:
            return
        ###safe check done
        food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = s
        pfood_dir_x, pfood_dir_y, padjoining_wall_x, padjoining_wall_y, padjoining_body_top, padjoining_body_bottom, padjoining_body_left, padjoining_body_right = s_prime


        alpha = self.C / (self.C + self.N[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right,a])
        ori_val = self.Q[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right,a]
        maxq = self.Q[pfood_dir_x, pfood_dir_y, padjoining_wall_x, padjoining_wall_y, padjoining_body_top, padjoining_body_bottom, padjoining_body_left, padjoining_body_right].max()
        

        self.Q[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right,a] = ori_val + alpha * (r + self.gamma * maxq-ori_val) 


    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        return utils.RIGHT

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 

        #Each state in the MDP is a tuple 
        #(food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, 
        #adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        #
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment;
        food_dir_x = 0
        if (food_x < snake_head_x):
            food_dir_x = 1
        if (food_x > snake_head_x):
            food_dir_x = 2

        food_dir_y = 0
        if (food_y < snake_head_y):
            food_dir_y = 1
        if (food_y > snake_head_y):
            food_dir_y = 2
        ###
        ### rock 2 * 1, record the left position
        adjoining_wall_x = 0
        if (snake_head_x == self.display_width - 2):
            adjoining_wall_x = 2
        if (snake_head_y == rock_y and (snake_head_x == rock_x - 1 or snake_head_x == rock_x)):
            adjoining_wall_x = 2
        if (snake_head_x == 1):
            adjoining_wall_x = 1
        if (snake_head_y == rock_y and (snake_head_x == rock_x + 1 or snake_head_x == rock_x + 2)):
            adjoining_wall_x = 1

        
        adjoining_wall_y = 0
        if (snake_head_y == self.display_height - 2):
            adjoining_wall_y = 2
        if (snake_head_y == rock_y - 1 and (snake_head_x == rock_x or snake_head_x == rock_x+1)):
            adjoining_wall_y = 2
        if (snake_head_y == 1):
            adjoining_wall_y = 1
        if (snake_head_y == rock_y + 1 and (snake_head_x == rock_x or snake_head_x == rock_x+1)):
            adjoining_wall_y = 1

        ### adjoin wall done
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        if (snake_head_x, snake_head_y - 1) in snake_body:
            adjoining_body_top = 1
        if (snake_head_x, snake_head_y + 1) in snake_body:
            adjoining_body_bottom = 1
        if (snake_head_x - 1, snake_head_y) in snake_body:
            adjoining_body_left = 1
        if (snake_head_x + 1, snake_head_y) in snake_body:
            adjoining_body_right = 1
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
