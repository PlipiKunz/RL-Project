
import gym
from gym.core import Env
import environment
import numpy as np

import globalVars
import keyboard

list_of_actions = []



def initialize_new_game(name, env, agent):
    """We don't want an agents past game influencing its new game, so we add in some dummy data to initialize"""
    
    env.reset()
    starting_frame = env.step(0)[0]
    
    dummy_action = 0
    dummy_reward = 0
    dummy_done = False
    for i in range(3):
        agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)

def make_env(name, agent):
    if name == "Bounce":
        env = environment.BounceEnv()
    else: 
        env = gym.make(name)
        
    return env

def good_actor_random(env):
    if np.random.random() < .1:
        return np.random.randint(low=0, high=3)
    else:
        return env.paddle_ball_relation()
    
def good_actor_epsilon(env):
    if globalVars.be_random:
        act =  np.random.randint(low=0, high=3)
    else:
        act =  env.paddle_ball_relation()
        
    return act
    
def get_user_action():
    if keyboard.is_pressed('up'):
        return 0
    elif keyboard.is_pressed('down'):
        return 2
    return 1

def get_actor_action(env):
    # env.render()
    
    # return good_actor_random(env)
    return good_actor_epsilon(env)
    # return good_actor_random(env)
    

def take_step(name, env, agent, score, debug, mode = "computer", learn = True, remember = True):
    
    #1 and 2: Update timesteps and save weights
    if learn:
        agent.total_timesteps += 1
        if agent.total_timesteps % 50000 == 0:
            agent.model.save_weights('recent_weights.hdf5')
            print('\nWeights saved!')

    #3: Take action
    prev_action = 0
    if len(list_of_actions) > 0:
        prev_action = list_of_actions[-1][2]
        
    next_frame, next_frames_reward, next_frame_terminal, info = env.step(prev_action)
    
    #4: Get next state
    new_state = next_frame
    
    #5: Get next action, using next state
    if mode == "computer":
        next_action = agent.get_action(new_state)
    else:
        next_action = get_actor_action(env)

    #6: If game is over,learn and then return the score
    if next_frame_terminal:
        if remember:
            list_of_actions.append([next_frame, next_frames_reward, next_action, next_frame_terminal])
            
            for mem in list_of_actions:
                agent.memory.add_experience(mem[0] , mem[1] , mem[2], mem[3])
            
            list_of_actions.clear()
            
        return (score + next_frames_reward),True , info  #(next_frames_reward),True
    
    #7: Now we add the next experience to memory
    if remember:
        list_of_actions.append([next_frame, next_frames_reward, next_action, next_frame_terminal])

    # 9: If the threshold memory is satisfied, make the agent learn from memory
    if len(agent.memory.frames) > agent.starting_mem_len and learn:

            agent.learn(debug)

    #8: If we are trying to debug this then render
    if debug:
        env.render()

    return (score + next_frames_reward),False, info  #(next_frames_reward), False

def play_episode(name, env, agent, debug = False, iterations = 0, mode = "computer", learn = True, remember = True):
    initialize_new_game(name, env, agent)
    done = False
    score = 0
    info = ""
    
    if np.random.rand() < globalVars.ga_epsilon:
        globalVars.be_random = True
    else:
        globalVars.be_random = False
        
    if globalVars.ga_epsilon > globalVars.ga_epsilon_min:
            globalVars.ga_epsilon -=globalVars.ga_epsilon_decay
            
            
    print(globalVars.ga_epsilon)
    while True:
        score,done,info  = take_step(name,env,agent,score, debug, mode, learn, remember)
        
        if done:
            break
    return score, info 