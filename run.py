import glob
import os
import sys
import time
import numpy as np

from utils import os_config, folder_creator

from enviroment.SimpleEnviroment import SimpleEnviroment 

import cv2

#from models.simplemodel import ActorCritic

from models.pytorch_models.A2C import ActorCritic


num_epochs = int( input( 'Ingrese la cantidad de iteraciones que desee implementar: ' ) )

min_values = [-1, -1]
max_values = [1, 1]

IM_WIDTH = 640
IM_HEIGHT = 480
SAVE_PATH = 'Saves/SimpleModel'
GAMMA = 0.3

folder_creator.recursive_folder( SAVE_PATH )
config = os_config.system_configuration()

actor_critic = ActorCritic( (3, IM_HEIGHT, IM_WIDTH), 2, min_values, max_values ,SAVE_PATH )



enviroment = SimpleEnviroment('model3', IM_WIDTH, IM_HEIGHT, 'Town02')
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


step_num = 0
best_reward = -1 * np.inf
for i in range( num_epochs ):

    is_saved_video = True if ( False ) else False
    rewards = []
    if i == 0:
        obs, _ ,done, _ = enviroment.start()
    else: 
        obs, _, done, _ = enviroment.reset()

    if is_saved_video:
        out = cv2.VideoWriter(f'Outputs/output-{ actor_critic.epochs + 1 }.avi',fourcc, 20.0, (640,480))
    
    while not done:
        action = actor_critic.get_action( obs / 255 )
        
        obs_next, reward, done, info = enviroment.step( action )
        rewards.append( reward )
        if is_saved_video:
            print_video( out, obs )
        obs = obs_next
        step_num += 1
        if done or step_num > 20: 
            step_num = 0
            max_reward = max( actor_critic.mean_rewards )
            best_reward = max_reward if max_reward > best_reward else best_reward
            actor_critic.learn( rewards, obs, done, GAMMA )
            rewards = []
        time.sleep( 1/20 )

    print( f'Iteración terminada \nepochs: {actor_critic.epochs}\nmax reward: {max_reward}\n mean: {np.mean(actor_critic.mean_rewards)}'  )
    
    if is_saved_video:
        out.release()

print( np.max( actor_critic.mean_rewards ), max_reward )

enviroment.destroy_actors()


