import gym
from gym import spaces

import cv2
import numpy as np


class BounceEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
            
        # There are 3 actions
        # 0 Move the paddle left
        # 1 Dont move the paddle 
        # 2 Move the paddle right
        self.action_space = spaces.Discrete(3)
        
        # The observations
        # 0 ball pos x,y
        # 1 ball velocity x,y
        # 2 paddle pos x, y top left
        self.observation_space = spaces.Discrete(6)
        
        # resets the space
        self.reset()
    
        self.set_obs()
    
    def step(self, action):
        self.paddle_update(action)
        done = self.ball_update()
        
        # reward = 10 * self.bounces
        reward = 0
        # if the ball is between the paddle height wise
        if(self.paddle_pos[1] <= self.ball_pos[1]) and (self.ball_pos[1] <= self.paddle_pos[1] + self.paddle_height):
            reward += 5
            
        # if the game is over, let ball hit own side wall
        if done:
            reward -= 25
            
        # if the game is done, you won
        if self.bounces > self.max_bounce:
            # reward = 10000
            done = True
        
        
        self.set_obs()
        return self.observation_space, reward, done, self.bounces
    
    def paddle_update(self, action):
        next_pos = self.paddle_pos
        
        # updates the position moving it up or down
        if(action == 0):
            next_pos[1] = next_pos[1] - 2
        elif (action == 1):
            pass
        elif (action == 2):
            next_pos[1] = next_pos[1] + 2
            
        
        # bound checks the paddle
        min_paddle_y = 0
        max_paddle_y = self.screen_height - self.paddle_height
        if(next_pos[1] < min_paddle_y):
            next_pos[1] = min_paddle_y
        elif(next_pos[1] > max_paddle_y):
            next_pos[1] = max_paddle_y
            
        self.paddle_pos = next_pos
    
    def ball_update(self):
        done = False
        
        next_ball_vel = self.ball_vel
        
        # if the ball has collided with the paddle
        if self.paddle_ball_collision():
            next_ball_vel = (-self.max_vel, self.ball_vel[1])
        
        
        # if the ball has made it into the goal we are done
        if self.ball_pos[0] >= self.screen_width:
            done = True
            
        # if the ball hits the far wall, set its x velocity 
        # positive, update score
        elif self.ball_pos[0] < 0:
            self.bounces += 1
            next_ball_vel = (self.max_vel, next_ball_vel[1])
            
            
        # if the ball bounces off the top or bottom wall
        # then set its y velocity going in the oposite direction
        if self.ball_pos[1] < 0:
            next_ball_vel = (next_ball_vel[0], self.max_vel)
        elif self.ball_pos[1] > self.screen_height:
            next_ball_vel = (next_ball_vel[0], -self.max_vel)
             
             
        # updates the balls posion and velocity
        self.ball_pos = (self.ball_pos[0] + next_ball_vel[0], self.ball_pos[1] + next_ball_vel[1])
        self.ball_vel = next_ball_vel
        
        return done
    
    def paddle_ball_collision(self):
        # if the ball is betwwen the paddle width wise
        if(self.paddle_pos[0] <= self.ball_pos[0]) and (self.ball_pos[0] <= self.paddle_pos[0]+ self.paddle_width):
            # if the ball is between the paddle height wise
            if(self.paddle_pos[1] <= self.ball_pos[1]) and (self.ball_pos[1] <= self.paddle_pos[1] + self.paddle_height):
                return True
        return False
    
    def reset(self):
        self.screen_width = 75
        self.screen_height = 50
        
        self.max_vel = 1
        
        # The position of the ball x,y
        self.ball_pos = [1,np.random.randint(0+1,self.screen_height-1)] #[1, self.screen_height//2]  
        # The velocity of the ball x,y
        self.ball_vel = [1, self.max_vel] #np.random.choice([self.max_vel,-self.max_vel])]
        
        
        # The width of the paddle
        self.paddle_width = 1
        # The height of the paddle
        self.paddle_height = 7
        
        # The position of the paddle
        self.paddle_pos = [self.screen_width - self.paddle_width - 2, self.screen_height // 2 - self.paddle_height//2]
        
        self.max_bounce = 10
        self.bounces = 0
        
        self.set_obs()
        
        return self.observation_space
        
    def set_obs(self):
        self.observation_space = np.array((self.ball_pos[0],self.ball_pos[1], self.ball_vel[0],  self.ball_vel[1],self.paddle_pos[1]))
        
    def render(self, mode="human", close = False):
        if(mode=="human"):
            
            cv2.namedWindow("bounce", cv2.WINDOW_NORMAL)
            screen = np.ones(shape=( self.screen_height+1,self.screen_width+1, 3))
            
            border_color = (0, 0,255)
            screen =cv2.line(screen, (0,0), (self.screen_width,0), border_color)
            screen =cv2.line(screen, (0,self.screen_height), (self.screen_width,self.screen_height), border_color)
            
            screen =cv2.line(screen, (0,0), (0,self.screen_width), border_color)
            screen =cv2.line(screen, (self.screen_width,0), (self.screen_width,self.screen_height), border_color)
            
            screen = cv2.circle(screen, center=(self.ball_pos[0], self.ball_pos[1]), radius=1, color=(255,255,0),thickness=-1)
            screen = cv2.rectangle(screen, self.paddle_pos, (self.paddle_pos[0] + self.paddle_width, self.paddle_pos[1] + self.paddle_height), (0,255,0),-1)
            
            cv2.imshow("bounce", screen)
            cv2.waitKey(1)