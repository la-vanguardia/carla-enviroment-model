import glob
import os
import sys
import time
import numpy as np

from utils import os_config, folder_creator

from enviroment.SimpleEnviroment import SimpleEnviroment 

import cv2

from models.simplemodel import ActorCritic

min_values = [-1, -1]
max_values = [1, 1]

IM_WIDTH = 640
IM_HEIGHT = 480
SAVE_PATH = 'Saves/SimpleModel'
GAMMA = 0.3

folder_creator.recursive_folder( SAVE_PATH )
config = os_config.system_configuration()


actor_critic = ActorCritic( (IM_HEIGHT, IM_WIDTH, 3), 2, min_values, max_values, False ,SAVE_PATH )

enviroment = SimpleEnviroment('model3', IM_WIDTH, IM_HEIGHT, config['map'])
fourcc = cv2.VideoWriter_fourcc(*config['codecc'])



def print_video( writter, image ):
    image = cv2.resize( image, (IM_WIDTH, IM_HEIGHT))
    position = ( 30,30 )
    cv2.putText(
    image, #numpy array on which text is written
    f"{ enviroment.vehicle.get_speed_limit()} ", #text
    position, #position at which writing has to start
    cv2.FONT_HERSHEY_SIMPLEX, #font family
    1,
    (255, 255, 255) ) 
    writter.write( image )

for i in range( 1000 ):
    rewards = []
    if i == 0:
        obs, _ ,done, _ = enviroment.start()
    else: 
        obs, _, done, _ = enviroment.reset()
    #out = cv2.VideoWriter(f'output-{time.time()}.avi',fourcc, 20.0, (640,480))
    
    while not done:
        action = actor_critic.get_action( obs / 255 )
        
        obs_next, reward, done, info = enviroment.step( action )
        rewards.append( reward )
        #print_video( out, obs )
        obs = obs_next
    break
    print( i )
    actor_critic.learn( rewards, obs, done, GAMMA )
    print( 'End Learn' )
    #out.release()

enviroment.destroy_actors()


