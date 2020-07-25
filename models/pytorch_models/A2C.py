from collections import namedtuple
import torch
import numpy as np
import json


import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from os.path import join, isfile

from models.pytorch_models.Actor import Actor
from models.pytorch_models.Critic import Critic 


Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])

class ActorCritic(mp.Process):

    _SAVE_FILE = 'actor-critic.json'

    def __init__( self, input_shape, output_shape, min_values, max_values, save_path=None ):
        self.actor = Actor( input_shape, output_shape )
        self.critic = Critic( input_shape, 1 )
        self.output_shape = output_shape
        self.min_values = min_values
        self.max_values = max_values

        self.trajectory = []
        self.epochs = 1
        self.mean_rewards = [  ]


        if save_path:
            self.file_path = join( save_path, self._SAVE_FILE )
            if isfile( self.file_path ):
                self.load(  )


    def policy( self, obs ):
        mu, sigma = self.actor( obs )
        value =  self.critic( obs )
        sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7 
        
        self.mu = mu.to(torch.device("cpu"))
        self.sigma = sigma.to(torch.device("cpu"))
        self.value = value.to(torch.device("cpu"))

        [mu[:,i].clamp_(float(self.min_values[i]), float(self.max_values[i])) for i in range(self.output_shape)]


        if len(self.mu.shape) == 0: #mu es un escalar
            self.mu.unsqueeze_(0) #evitará que la multivariante noral de un error
        action_distribution = MultivariateNormal(self.mu, torch.eye(self.output_shape) * self.sigma, validate_args = True)

        return action_distribution

    def preprocess_obs(self, obs):
        obs = np.array(obs)
        if len(obs.shape) == 3:
            obs = np.reshape(obs, (obs.shape[2], obs.shape[1], obs.shape[0]))
        
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        return obs

    def process_action(self, action):
        [action[:,i].clamp_(float(self.min_values[i]), float(self.max_values[i])) for i in range(self.output_shape)]
            
        action = action.to(torch.device("cpu"))
        return action.numpy().squeeze(0)

    def get_action(self, obs):
        obs = self.preprocess_obs(obs)
        action_distribution = self.policy(obs)
        value = self.value
        
        action = action_distribution.sample()
        log_prob_a = action_distribution.log_prob(action)
        action = self.process_action(action)
        
        if action[0] < 0:
            action = np.append( action, -1 * action[0] )
            action[0] = 0
        else:
            action = np.append( action, 0.0 )

        self.trajectory.append(Transition(obs, value, action, log_prob_a))
        return action

    def calculate_n_step_return(self, n_step_rewards, final_state, done, gamma):
        """
        Calcula el valor de retorno dados n-pasos para cada uno de los estados de entrada
        :param n_step_rewards: Lista de las recompensas obtenidas en cada uno de los n estados
        :param final_state: Estado final tras las n iteraciones
        :param done: Variable booleana con valor True si se ha alcanzado el estado final del entorno
        :param gamma: Factor de Descuento para el cálculo de la diferencia temporal.
        :return: El valor final de cada estado de los n ejecutados
        """
        g_t_n_s = list();
        with torch.no_grad():
            g_t_n = torch.tensor([[0]]).float() if done else self.critic(self.preprocess_obs(final_state)).cpu()
            for r_t in n_step_rewards[::-1]:
                g_t_n = torch.tensor(r_t).float() + gamma * g_t_n
                g_t_n_s.insert(0, g_t_n)
            return g_t_n_s


    def calculate_loss(self, trajectory, td_targets):
        """
        Calcular la pérdida del crítico y del actor utilizando los td_targets y la trayectoria por otro
        :param trajectory:
        :param td_targets:
        :return:
        """
        n_step_trajectory = Transition(*zip(*trajectory))
        v_s = n_step_trajectory.value_s
        log_prob_a = n_step_trajectory.log_prob_a
        actor_losses = []
        critic_losses = []
        
        for td_target, critic_prediction, log_p_a in  zip(td_targets, v_s, log_prob_a):
            td_error = td_target - critic_prediction
            actor_losses.append(- log_p_a * td_error) # td_error es un estimador insesgado de Advantage
            critic_losses.append(F.smooth_l1_loss(critic_prediction, td_target))
            
            

        actor_loss = torch.stack(actor_losses).mean()
            
        critic_loss = torch.stack(critic_losses).mean()
        
        return actor_loss, critic_loss
            

    def learn(self, rewards ,n_th_observation, done, gamma):
        td_targets = self.calculate_n_step_return(rewards, n_th_observation, done, gamma)
        actor_loss, critic_loss = self.calculate_loss(self.trajectory, td_targets)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 0.01)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 0.01)
       
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.trajectory.clear()
        
        self.epochs += 1
        self.mean_rewards.append( np.mean( rewards ) )



    def save( self ):
        safe_data = {
            'epochs': self.epochs,
            'mean rewards': self.mean_rewards
        }

        safe_data['critic'] = self.critic.save()
        safe_data['actor'] = self.actor.save()
        
        with open( self.file_path, 'w' ) as json_file:
            json.dump( safe_data, json_file )

    def load( self ):
        with open( self.file_path, 'r' ) as json_file:
            safe_data = json.load( json_file )

        self.mean_rewards = safe_data['mean rewards']
        self.epochs = safe_data['epochs']
        self.critic.load( safe_data['critic'] )
        self.actor.load( safe_data['actor'] )

