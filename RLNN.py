import tensorflow as tf
import keras
import numpy as np
import tensorflow_probability as tfp
import time 
import threading
from math import pi, sin, cos


from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import activations
from tensorflow.keras.layers import Flatten, Conv1D, Reshape, Conv1DTranspose, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam

tfd = tfp.distributions

class Model_creater(tf.keras.Model):
    def __init__(self, date, state):
        super().__init__()
        self.date = date
        self.n_actions = 2
        self.n_pairs = state
        self.f_n = int(self.n_pairs * self.n_actions)
        self.action_per_episode = 120
        self.pos_log_std = tf.Variable(-1.7, trainable=True, dtype=tf.float32) #-0.7
        self.z_log_std = tf.Variable(-1.7, trainable=True, dtype=tf.float32)
        self.pos_bound = 300  
        self.z_bound = 0.5 
        self.clip_ratio = 0.2
        self.actor_optimizer = Adam(learning_rate=0.0002)
        self.critic_optimizer = Adam(learning_rate=0.0004)
        self.pos_std_optimizer = Adam(learning_rate=0.008)
        self.z_std_optimizer = Adam(learning_rate=0.008)
        self.old_actor, _ = self.create_actor_critic()
    
    def change_date(self, date):
        self.date = date
 
    def create_actor_critic(self):
        inputs1 = tf.keras.Input(shape=(self.n_pairs, self.n_actions))               
        x = tf.keras.layers.LSTM(128, return_sequences=False,
                  name = 'lstm1')(inputs1)
        x = tf.keras.layers.Dense(64, activation = 'tanh')(x)
        out = Dense(1,name='value')(x) 
        self.critic = tf.keras.Model(inputs1, out) 
        inputs = tf.keras.Input(shape=(self.n_pairs, self.n_actions))        
        x = tf.keras.layers.LSTM(64, return_sequences=False,
                  name = 'lstm1')(inputs)   
        x = tf.keras.layers.Dense(32, activation = 'tanh')(x)
        action_mean = Dense(self.n_actions, activation = 'tanh', name='action_mean')(x)
        value_actor = Dense(1, name="V_mean")(x)
        self.actor = tf.keras.Model(inputs=inputs, outputs=[action_mean, value_actor]) 
        return self.actor, self.critic
    
    
    def pre_action_maker(self, cell_state=0, random_start=True): # 第一列是位置，第二列是速度
        self.whole_pre_decision = np.zeros([self.action_per_episode + 2*self.n_pairs, self.n_actions])
        self.whole_pre_decision_for_state = np.zeros([self.action_per_episode + 2*self.n_pairs, self.n_actions])
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_prob_u = []
        self.episode_entropy = []
        if random_start:
            for i in range(self.n_pairs):
                random_A = np.power(-1, i) * np.random.randint(80, 230)
                random_z0 = np.random.rand(1)
                self.whole_pre_decision[i,0] = random_A
                self.whole_pre_decision[i+self.n_pairs,0] = random_A
                self.whole_pre_decision[i,1] = random_z0
                self.whole_pre_decision[i+self.n_pairs,1] = random_z0
                if random_A > 0:
                    self.whole_pre_decision_for_state[i,0] = (random_A-80) / 75
                    self.whole_pre_decision_for_state[i+self.n_pairs,0] = (random_A-80) / 75
                else:
                    self.whole_pre_decision_for_state[i,0] = (random_A + 80) / 75
                    self.whole_pre_decision_for_state[i+self.n_pairs,0] = (random_A+80) / 75    
                self.whole_pre_decision_for_state[i,1] = random_z0
                self.whole_pre_decision_for_state[i+self.n_pairs,1] = random_z0
        else:
            for i in range(cell_state.shape[0]):
                if cell_state[i, 0] >= 0:
                    self.whole_pre_decision[i, 0] = np.array(75 * cell_state[i, 0] + 80, dtype='int32')
                    self.whole_pre_decision[i + self.n_pairs, 0] = np.array(75 * cell_state[i, 0] + 80, dtype='int32')
                else:
                    self.whole_pre_decision[i, 0] = np.array(75 * cell_state[i, 0] -80, dtype='int32')
                    self.whole_pre_decision[i + self.n_pairs, 0] = np.array(75 * cell_state[i, 0] - 80, dtype='int32')                    

            self.whole_pre_decision[:self.n_pairs, 1] = (cell_state[:, 1] + 1) /2   
            self.whole_pre_decision[self.n_pairs:2*self.n_pairs, 1] = (cell_state[:, 1] + 1) /2
            self.whole_pre_decision_for_state[:self.n_pairs, : ] = cell_state
            self.whole_pre_decision_for_state[self.n_pairs:2*self.n_pairs, :] = cell_state
        if  self.whole_pre_decision_for_state[0, 0]>= 0:
            Zhengkaitou = 0
        else:
            Zhengkaitou = 1            
        for i in range(self.action_per_episode):
            state = np.zeros([1, self.f_n]) 
            state_for_state = np.zeros([1, self.f_n]) 
            for j in range(self.n_pairs):
                state[0, j*2 :2*j+2] = self.whole_pre_decision[self.n_pairs+j+i, :]
                state_for_state[0, j*2 :2*j+2] = self.whole_pre_decision_for_state[self.n_pairs+j +i, :]                
            state = state.reshape(1, self.n_pairs, self.n_actions)
            state_for_state = state_for_state.reshape(1, self.n_pairs, self.n_actions)            
            self.episode_states.append(state_for_state)
            actions, log_prob_u, entropy = self.act(state_for_state)            
            self.episode_log_prob_u.append(log_prob_u)
            self.episode_entropy.append(entropy)
            self.episode_actions.append(actions)        
            pos_action = (actions[0, 0] + 1)* self.pos_bound/4 + 80
            pos_action = np.clip(pos_action, 80 , 300)              
            z0_action = (actions[0, 1] + 1) * self.z_bound
            self.whole_pre_decision_for_state[i+2*self.n_pairs, 0] = np.power(-1, i+Zhengkaitou) * (actions[0, 0] + 1) #(-2,2)
            if self.whole_pre_decision_for_state[i+2*self.n_pairs, 0] > 0:
                self.whole_pre_decision_for_state[i+2*self.n_pairs, 0] = self.whole_pre_decision_for_state[i+2*self.n_pairs, 0]
            elif self.whole_pre_decision_for_state[i+2*self.n_pairs, 0] < 0:
                self.whole_pre_decision_for_state[i+2*self.n_pairs, 0] = self.whole_pre_decision_for_state[i+2*self.n_pairs, 0]
            elif self.whole_pre_decision_for_state[i+2*self.n_pairs, 0] == 0 and self.whole_pre_decision_for_state[i+2*self.n_pairs-1, 0] < 0:
                self.whole_pre_decision_for_state[i+2*self.n_pairs, 0] = 0.1
            else:
                self.whole_pre_decision_for_state[i+2*self.n_pairs, 0] = -0.1
            self.whole_pre_decision_for_state[i+2*self.n_pairs, 1] = actions[0, 1] #(-1,1)       
            self.whole_pre_decision[i+2*self.n_pairs:, 0] = int(np.power(-1, i+Zhengkaitou) * pos_action)
            self.whole_pre_decision[i+2*self.n_pairs:, 1] = z0_action
        return  self.whole_pre_decision, \
                np.array(self.episode_states).reshape(self.action_per_episode,self.n_pairs,self.n_actions),\
                np.array(self.episode_actions).reshape(-1, self.n_actions), \
                np.array(self.episode_log_prob_u).reshape(-1, 1), np.array(self.episode_entropy).reshape(-1, 1)
                
    def act(self, state, test=False, use_random=False):    
        if use_random:
            a = tf.random.uniform(shape=(1, self.n_actions), minval=-1, maxval=1, dtype=tf.float64)
        else:
            means, _ = self.actor(state)
            a, entropy, log_prob_u = self.process_actions(means, test=test)
        a = np.array(a)
        actions = np.clip(a, -1, 1)
        return actions, np.array(log_prob_u)[0, ], np.array(entropy)
            
    def process_actions(self, means, test=False, eps=1e-6):
        raw_actions = means
        pos_std = tf.exp(self.pos_log_std)
        z_std = tf.exp(self.z_log_std)
        if not test:
            raw_actions = tfd.Normal(loc=means, scale=[pos_std, z_std]).sample()
        log_prob_u = tfd.Normal(loc=means, scale=[pos_std, z_std]).log_prob(raw_actions) 
        log_prob_u = tf.reduce_sum(log_prob_u, axis=-1) 
        entropy = tfd.Normal(loc=means, scale=[pos_std, z_std]).entropy()
        entropy = tf.reduce_sum(entropy, axis=-1) 
        return raw_actions, entropy, log_prob_u #a, entrophy

    def decoder_omega_maker(self):
        u = 0.077
        A = 0.16
        self.omega = np.zeros([self.action_per_episode+self.n_pairs, 700])
        for i in range(self.action_per_episode+self.n_pairs):
            z_0 = self.whole_pre_decision[i, 1]
            distance = abs(self.whole_pre_decision[i, 0] \
                           - self.whole_pre_decision[i-1, 0])
            A2 = int(distance) 
            theta = (A2 / 4096) * 360 / 2 
            A0 = 2 * A * np.sin(theta * np.pi / 180)
            rescaled_st = ((0.3/230) * (A2-220)) + 0.45 + (0.1 + (0.4/330)* (A2-220)) *z_0 
            rescaled_st = np.clip(rescaled_st, 0.3, 1.0)
            f = rescaled_st * u / A0
            w = 2 * pi * f
            omega_instant = np.zeros(700)
            for j in range(0, A2):
                t = (1/w) * np.arccos(1 - (j)/(0.5*A2))
                omega_instant[j] = 0.5*A2*w*sin(w*t)
            self.omega[i,:int(distance)] = omega_instant[0:int(distance)]
        self.omega = np.array(self.omega, dtype = 'int32')
        return self.omega
    
    def policy_learn(self, batch_states, batch_actions, batch_log_probs, batch_gaes, batch_target_values):
        with tf.GradientTape(persistent=True) as tape:
            pos_std = tf.exp(self.pos_log_std)
            z_std = tf.exp(self.z_log_std)
            means, _ = self.actor(batch_states)
            state_values = self.critic(batch_states)
            dist = tfd.Normal(loc=means, scale=[pos_std, z_std])
            new_log_probs = dist.log_prob(batch_actions)
            new_log_probs = tf.reduce_sum(new_log_probs, axis=-1)
            entropy = dist.entropy()            
            new_log_probs = tf.reshape(new_log_probs,[-1,1])
            ratios = tf.exp(new_log_probs - batch_log_probs)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                              clip_value_max=1+self.clip_ratio)
            loss_clip = tf.minimum(batch_gaes * ratios, batch_gaes * clipped_ratios)
            loss_clip = tf.reduce_mean(loss_clip)
            entropy = tf.reduce_mean(entropy)
            vf_loss = tf.reduce_mean(tf.math.square(state_values - batch_target_values))
            total_loss = -loss_clip
        train_variables = self.actor.trainable_variables[0:7]   
        grad = tape.gradient(total_loss, train_variables)  
        self.actor_optimizer.apply_gradients(zip(grad, train_variables))  
        grad = tape.gradient(vf_loss, self.critic.trainable_variables)  
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_variables)) 
        grad = tape.gradient(total_loss, [self.pos_log_std])
        self.pos_std_optimizer.apply_gradients(zip(grad, [self.pos_log_std])) 
        grad = tape.gradient(total_loss, [self.z_log_std])
        self.z_std_optimizer.apply_gradients(zip(grad, [self.z_log_std])) 
    
    def aux_learn(self, batch_states, batch_target_values):# need: state, episode_value, old_pi, gaes       
        old_pos_std = tf.exp(self.pos_log_std)
        old_z_std = tf.exp(self.z_log_std)        
        pos_std = tf.exp(self.pos_log_std)
        z_std = tf.exp(self.z_log_std)
        with tf.GradientTape(persistent=True) as tape:
            means, actor_state_values = self.actor(batch_states)
            old_means, _ = self.old_actor(batch_states)
            state_values = self.critic(batch_states)
            dist_new = tfd.Normal(loc=means, scale=[pos_std, z_std])
            dist_old = tfd.Normal(loc=old_means, scale=[old_pos_std, old_z_std])
            kl = tf.reduce_mean(tfd.kl_divergence(dist_old, dist_new))
            L_joint = 0.5 * tf.reduce_mean(tf.math.square(actor_state_values - batch_target_values)) + 1*kl # 1 refer to beta
            L_value = tf.reduce_mean(tf.math.square(state_values - batch_target_values)) 
        grad = tape.gradient(L_joint, self.actor.trainable_variables)  # compute gradient
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_variables))  
        grad = tape.gradient(L_value, self.critic.trainable_variables)  # compute gradient
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_variables))
    def update_old_actor(self, tau):
        weights = self.actor.get_weights()
        target_weights = self.old_actor.get_weights()
        for i in range(len(target_weights)):  # set tau% of target model to be new weights
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        self.old_actor.set_weights(target_weights)
                
    def test(self):
        return self.policy_log_std
    
    def save_model(self, index, fail_times):
        self.actor.save_weights('Model/%s_%d_%d_actor.h5'%(self.date, index, fail_times))
        self.critic.save_weights('Model/%s_%d_%d_critic.h5'%(self.date, index, fail_times))
        actor_opt_weigthts = self.actor_optimizer.get_weights()
        critic_opt_weigthts = self.critic_optimizer.get_weights()
        pos_opt_weigthts = self.pos_std_optimizer.get_weights()
        z_opt_weigthts = self.z_std_optimizer.get_weights()
        np.save('Model/opt/%s_%d_%d_a_opt.npy'%(self.date, index, fail_times),np.array(actor_opt_weigthts , dtype = 'object'))
        np.save('Model/opt/%s_%d_%d_c_opt.npy'%(self.date, index, fail_times),np.array(critic_opt_weigthts , dtype = 'object'))
        np.save('Model/opt/%s_%d_%d_pos_opt.npy'%(self.date, index, fail_times),np.array(pos_opt_weigthts , dtype = 'object'))
        np.save('Model/opt/%s_%d_%d_z_opt.npy'%(self.date, index, fail_times),np.array(z_opt_weigthts , dtype = 'object'))
        pos_log_std_np = self.pos_log_std.numpy()
        z_log_std_np = self.z_log_std.numpy()       
        np.save('Model/std/%s_%d_%d_pos_log_std.npy'%(self.date, index, fail_times), pos_log_std_np)
        np.save('Model/std/%s_%d_%d_z_log_std.npy'%(self.date, index, fail_times), z_log_std_np)
        print('\nModel_saved\n')
        
    def load_models(self, index, fail_times, load_std=True):
        self.actor.load_weights('Model/%s_%d_%d_actor.h5'%(self.date, index, fail_times))
        self.critic.load_weights('Model/%s_%d_%d_critic.h5'%(self.date, index, fail_times))
        if load_std == True:
            pos_log_std_np = np.load('Model/std/%s_%d_%d_pos_log_std.npy'%(self.date, index, fail_times))
            z_log_std_np = np.load('Model/std/%s_%d_%d_pos_log_std.npy'%(self.date, index, fail_times))    
            self.pos_log_std.assign(pos_log_std_np)
            self.z_log_std.assign(z_log_std_np)
        
    def load_opt(self, index, fail_times):
        actor_opt_weigthts = np.load('Model/opt/%s_%d_%d_a_opt.npy'%(self.date, index, fail_times), allow_pickle = True)
        critic_opt_weigthts = np.load('Model/opt/%s_%d_%d_c_opt.npy'%(self.date, index, fail_times), allow_pickle = True)
        pos_opt_weigthts = np.load('Model/opt/%s_%d_%d_pos_opt.npy'%(self.date, index, fail_times), allow_pickle = True)
        z_opt_weigthts = np.load('Model/opt/%s_%d_%d_z_opt.npy'%(self.date, index, fail_times), allow_pickle = True)
        self.actor_optimizer.set_weights(actor_opt_weigthts)
        self.critic_optimizer.set_weights(critic_opt_weigthts)
        self.pos_std_optimizer.set_weights(pos_opt_weigthts)
        self.z_std_optimizer.set_weights(z_opt_weigthts)

    def cal_expert_log_prob(self, episode_pre_states, episode_actions):
        raw_actions = episode_actions
        means, _ = self.actor(episode_pre_states)
        pos_std = tf.exp(self.pos_log_std)
        z_std = tf.exp(self.z_log_std)  
        log_prob_u = tfd.Normal(loc=means, scale=[pos_std, z_std]).log_prob(raw_actions) 
        log_prob_u = tf.reduce_sum(log_prob_u, axis=-1)     
        return log_prob_u

    
    
    
    
    
    
    
    
    