import agent
import environment_methods 

import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np


name = 'Bounce'
agent = agent.Agent(possible_actions=[0,1,2], starting_mem_len=50_000, max_mem_len=750_000, starting_epsilon=1, learn_rate=.00025)
env = environment_methods.make_env(name, agent)

avg_scores = [-1]
score_list = []
duration = []
bounces = []

scores = deque(maxlen=100)
max_score = -1
max_bounces = -1

env.reset()
env.render()

for i in range(1_000):
    print('\nEpisode: ' + str(i))
    environment_methods.play_episode(name, env, agent, False, i, "human", False, True)

for i in range(1_000):
    timesteps = agent.total_timesteps
    start_time = time.time()

    score , info = environment_methods.play_episode(name, env, agent, False, i, "computer",True, False)
    
    scores.append(score)

    if score > max_score:
        max_score = score
        
    if info > max_bounces:
        max_bounces = info

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_timesteps - timesteps))
    print('Duration: ' + str(time.time() - start_time))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))
    print('Bounces: ' + str(info))
    print('Max Bounces: ' + str(max_bounces))
    print('Epsilon: ' + str(agent.epsilon))

    score_list.append(score)
    duration.append(agent.total_timesteps - timesteps)
    bounces.append(info)
    
    if (i%50==0) or (i >= 150 and i%10==0) :
        plt.plot(np.arange(0,i+1,1),score_list)
        plt.title("Scores")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.savefig(fname =  f"plots/scores/scores_at_iteration_{i}.png")
        plt.close()

        plt.plot(np.arange(0,i+1,1),duration)
        plt.title("Durations")
        plt.xlabel("Iteration")
        plt.ylabel("Duration (s)")
        plt.savefig(fname =  f"plots/durations/durations_at_iteration_{i}.png")
        plt.close()
        
        plt.plot(np.arange(0,i+1,1),bounces)
        plt.title("Bounces")
        plt.xlabel("Iteration")
        plt.ylabel("Bounces")
        plt.savefig(fname =  f"plots/bounces/bounces_at_iteration_{i}.png")
        plt.close()

    if i%50==0 and i!=0:
        avg_scores.append(sum(scores)/len(scores))
        plt.plot(np.arange(0,i+1,50), avg_scores)
        plt.title("Average Scores")
        plt.xlabel("Iteration")
        plt.ylabel("Average Score")
        plt.savefig(fname =  f"plots/average_scores/ave_scores_at_itteration_{i}.png")
        plt.close()

