import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import copy
from RLNN import Model_creater
from tensorflow.keras.models import load_model
tfd = tfp.distributions
tf.keras.backend.set_floatx('float32')

from Env import Fish_Env
reward_length = 8
state_length = reward_length + 8
env = Fish_Env(2025, reward_length, state_length, -0.1563081, 0.098246, 60, -0.000726) 
def cal_gaes(episode_rewards, value_his, lamda):
    deltas = np.zeros([batch_size, 1]) 
    for i in range(batch_size-1):
        deltas[i,0] = episode_rewards[i] + gamma * value_his[i+1, 0] - value_his[i, 0]
    gaes = copy.deepcopy(deltas) 
    for t in reversed(range(batch_size-2)):
        gaes[t,0] = gaes[t,0] + lamda * gamma * gaes[t+1,0] 
    return gaes

date = 'date' 
gamma = 0.99
MC =  Model_creater(date, state_length)
actor, critic = MC.create_actor_critic()
reward_his = []
entropy_his = []

fail_times = 0
# MC.load_models(150, 0, True)
last_state = 0 

wpd_for_learning_path = []
for episodes in range(501):
    if episodes % 250 == 0:
        measure = True
    else:
        measure = False
    if episodes >= 1:                
        whole_pre_decision, episode_states, episode_actions, episode_log_prob_u, \
            episode_entropy = MC.pre_action_maker(last_state, False)
    else:    
        whole_pre_decision, episode_states, episode_actions, episode_log_prob_u, \
            episode_entropy = MC.pre_action_maker() 
    episode_pre_states = episode_states
    episode_pre_actions = episode_actions
    omega = MC.decoder_omega_maker()
    wpd_for_learning_path.append(whole_pre_decision)
    wpd_for_learning_path_array = np.array(wpd_for_learning_path)
    episode_rewards, error, P, if_sensor_change_list = env.step(whole_pre_decision, omega, measure)
    if error:
        reward_his.append(np.array([0]))
        entropy_his.append(np.array([0]))
        continue 
    else:
        batch_size = episode_rewards.shape[0]
        episode_states = episode_states[: batch_size, :] 
        last_state = episode_states[-1,:,:].reshape(episode_states.shape[1], episode_states.shape[2])
        episode_actions = episode_actions[: batch_size, ]
        episode_log_prob_u = episode_log_prob_u[: batch_size, ]
        episode_entropy = episode_entropy[: batch_size, ]
        episode_values = critic(episode_states).numpy()
        episode_gaes = cal_gaes(episode_rewards, episode_values, 0.9)
        episode_next_v = np.r_[episode_values[1:], np.zeros([1,1])]        
        episode_target_values = episode_gaes + episode_values
        reward_his.append(sum(episode_rewards)/batch_size)
        entropy_his.append(sum(episode_entropy)/batch_size)       
        episode_states = episode_states[0:batch_size-1, :] # 最后一布target_v有问题，不能要
        episode_actions = episode_actions[0:batch_size-1, :]
        episode_log_prob_u = episode_log_prob_u[0:batch_size-1, :]
        episode_gaes = episode_gaes[0:batch_size-1, :]
        episode_target_values = episode_target_values[0:batch_size-1, :]               
        for i in range(2):
            MC.policy_learn(episode_states, episode_actions, episode_log_prob_u, episode_gaes, episode_target_values)
        MC.update_old_actor(tau=1.0)
        for i in range(2):
            MC.aux_learn(episode_states, episode_target_values)        
        # if episodes == 0:
        #     MC.load_opt(150, 0)
        if episodes % 10 == 0 and episodes != 0:
            acc_reward = []
            roll_reward = 0
            for i in range(len(reward_his)):
                roll_reward = roll_reward + reward_his[i]
                acc_reward.append(roll_reward)        
            plt.plot(acc_reward)
            plt.title('acc_reward')
            plt.show()
            plt.plot(entropy_his)
            plt.title('entropy')
            plt.show()
            save_reward_his = np.array(reward_his, dtype = object)
            save_entropy_his = np.array(entropy_his, dtype = object)
            np.save('Reward_his/entropy_%s_%d_%d.npy'%(date, episodes, fail_times), np.array(save_entropy_his))
            np.save('Reward_his/reward_%s_%d_%d.npy'%(date, episodes, fail_times), np.array(save_reward_his))
            MC.save_model(episodes, fail_times)
            # env.power_cut(60)  