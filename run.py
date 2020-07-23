import glob
import os
import sys
import time
import numpy as np

from enviroment.SimpleEnviroment import SimpleEnviroment 

import cv2

from models.simplemodel import ActorCritic

min_values = [0, -1, 0]
max_values = [1, 1, 1]

IM_WIDTH = 640
IM_HEIGHT = 480

actor_critic = ActorCritic( (IM_HEIGHT, IM_WIDTH, 3), 3, min_values, max_values)

enviroment = SimpleEnviroment('model3', IM_WIDTH, IM_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

gamma = 0.3

def print_video( writter, image ):
    image*= 255
    
    image = image.astype( int )
    image = cv2.resize( image, (IM_WIDTH, IM_HEIGHT))
    position = ( 30,30 )
    cv2.putText(
    image, #numpy array on which text is written
    f"{ enviroment.vehicle.get_speed_limit()} ", #text
    position, #position at which writing has to start
    cv2.FONT_HERSHEY_SIMPLEX, #font family
    1,
    (255, 0, 0) ) 
    out.write( image )

for i in range( 1000 ):
    rewards = []
    #uts/output-{time.time()}.avi',fourcc, 20.0, (640,480))
    obs, _ ,done, _ = enviroment.start()
    while not done:
        action = actor_critic.get_action( obs )
        obs_next, reward, done, info = enviroment.step( action )
        rewards.append( reward )
        time.sleep( 1/20 )
        #print_video( out, obs )
        obs = obs_next
    print( i )
    actor_critic.learn( rewards, obs, done, gamma )
    print( 'End Learn' )
    #out.release()

enviroment.destroy_actors()


