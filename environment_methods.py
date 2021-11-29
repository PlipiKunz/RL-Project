
import gym
import environment
import numpy as np


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

def take_step(name, env, agent, score, debug):
    
    #1 and 2: Update timesteps and save weights
    agent.total_timesteps += 1
    if agent.total_timesteps % 50000 == 0:
      agent.model.save_weights('recent_weights.hdf5')
      print('\nWeights saved!')

    #3: Take action
    next_frame, next_frames_reward, next_frame_terminal, info = env.step(agent.memory.actions[-1])
    
    #4: Get next state
    new_state = next_frame
    
    #5: Get next action, using next state
    next_action = agent.get_action(new_state)

    #6: If game is over, return the score
    if next_frame_terminal:
        agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)
        return (score + next_frames_reward),True , info  #(score + next_frames_reward),True

    #7: Now we add the next experience to memory
    agent.memory.add_experience(next_frame, next_frames_reward, next_action, next_frame_terminal)

    #8: If we are trying to debug this then render
    if debug:
        env.render()

    # 9: If the threshold memory is satisfied, make the agent learn from memory
    if len(agent.memory.frames) > agent.starting_mem_len:
        agent.learn(debug)

    return (score + next_frames_reward),False, info  #(score + next_frames_reward), False

def play_episode(name, env, agent, debug = False, iterations = 0):
    initialize_new_game(name, env, agent)
    done = False
    score = 0
    info = ""
    while True:
        score,done,info  = take_step(name,env,agent,score, debug)
        
        if iterations%100 == 0:
            env.render()
            
        if done:
            break
    return score, info 