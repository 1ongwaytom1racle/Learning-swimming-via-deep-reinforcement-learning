import numpy as np
import time
import threading
import pyvisa
from New_Sensor import Sensor
from SCS_Control import SCS_Control
Move = SCS_Control().Move  #position,speed,acc
rm = pyvisa.ResourceManager()
device = rm.open_resource('USB0::0xF4EC::0xF4EC::SPD13DCX6R1132::INSTR')

class Fish_Env():
    def __init__(self, mid_pos, average_scale, state_length, fup, fdown, sample_time, M_0):
        super().__init__()
        self.average_scale = average_scale
        self.mid_pos = mid_pos
        self.sample_rate = 80
        self.sample_time = sample_time
        self.sample_points = int(self.sample_rate * self.sample_time)
        self.delta_time = 1/self.sample_rate
        self.fup = fup
        self.fdown = fdown
        self.M_0 = M_0
        self.prepare_length = state_length
        self.go_length = 2 * self.prepare_length - average_scale - 1 #假设S=2，r=1，则需要 s+s-r-1=2步时就开始采集，因为Time——window_list不记录第一个排尾
        self.If_sensor_change_list = []

    def Data_refinement(self):
        list_index = 0
        refine_index = 0
        self.refined_data = np.zeros([160000, 2])
        start_recorder = []
        while list_index <= (len(self.pos_omega_his) - 1):
            roll_index = 0
            if list_index != 0:
                time_new_start = self.time_list_for_refine[list_index - 1]
            else:
                time_new_start = 0
            while roll_index <= (len(self.pos_omega_his[list_index]) - 2):      
                if self.time_omega_his[list_index][roll_index] != self.time_omega_his[list_index][roll_index + 1]:
                    # 第一个只放new_start
                    if roll_index == 0:                    
                        self.refined_data[refine_index, 0] = time_new_start
                        self.refined_data[refine_index, 1] = self.pos_omega_his[list_index][roll_index]
                        refine_index += 1
                        roll_index += 1
                    else:
                        self.refined_data[refine_index, 0] = time_new_start + self.time_omega_his[list_index][roll_index - 1]
                        self.refined_data[refine_index, 1] = self.pos_omega_his[list_index][roll_index - 1]
                        refine_index += 1
                        roll_index += 1
                else:
                    roll_index += 1
                if roll_index == (len(self.pos_omega_his[list_index]) - 1):
                    start_recorder.append([time_new_start,refine_index])
                    break
            list_index += 1         
        refine_index -= 1
        omega = np.zeros([refine_index,2])
        for i in range(1, refine_index):
            delta_theta = 2 * 0.001533*(self.refined_data[i+1, 1] - self.refined_data[i-1, 1]) #2 * 0.001533~ 4096 -> 2*pi
            delta_time = self.refined_data[i+1, 0] - self.refined_data[i-1, 0]
            omega[i,0] = self.refined_data[i,0]
            omega[i,1] = delta_theta / delta_time
        x = omega[:, 0]
        y = omega[:, 1]
        x2 = np.linspace(0, self.sample_time-self.delta_time, self.sample_points)
        y2 = np.interp(x2, x, y, 0, 0)
        self.omega = y2
        x = self.refined_data[:refine_index, 0]
        y = self.refined_data[:refine_index, 1]
        x2 = np.linspace(0, self.sample_time-self.delta_time, self.sample_points)
        theta = np.interp(x2, x, y, 0, 0)
        self.costheta = np.array(theta).reshape(-1, 1)
        self.sintheta = np.array(theta).reshape(-1, 1)
        theta = np.array(theta).reshape(-1, 1)
        self.omega = self.omega[1:]        
        return self.refined_data
    
    def reward_cal(self):
        data = Sensor().sensor_data()
        cal_data = np.zeros([data.shape[0], 6])
        cal_data[:, 0] = 4* (data[:, 0] - self.M_0)   
        cal_data[:, 1] = data[:, 1]
        cal_data[:, 2] = data[:, 2]
        cal_data[:, 4] = self.omega[:]
        for i in range(cal_data.shape[0]):
            cal_data[i, 3] = 20 * (cal_data[i, 2] - cal_data[i, 1] - \
                                   (self.fup - self.fdown))
            cal_data[i, 5] = cal_data[i, 0] * cal_data[i ,4]
        self.time_window_list = []
        for j in range(len(self.time_list_for_refine)):

            if j >= self.average_scale:
                roll_index = [int(self.sample_rate*self.time_list_for_refine[j-self.average_scale]), 
                              int(self.sample_rate*self.time_list_for_refine[j])]
                roll_data = cal_data[roll_index[0]:roll_index[1]]
                if roll_data.shape[0] != 0: #避免3——23出现空的time_window_list报错
                    self.time_window_list.append(roll_data)
                    

        return self.time_window_list, cal_data

    def quick_sensor_check(self):
        M_0, fup, fdown = Sensor().quick_sensor_check()  
        self.M_0 = M_0 
        self.fup = fup
        self.fdown = fdown
        self.If_sensor_change_list.append(np.array([self.M_0, self.fup, self.fdown]))
        return self.fup, self.fdown
        
    def interaction_with_env(self, pre_pos, pre_omega):
        self.pos_omega_his = []
        self.time_omega_his = []
        self.time_list_for_refine = []
        real_Move_pos = pre_pos[:, 0] + self.mid_pos
        real_Move_pos = np.array(real_Move_pos, dtype='int32')
        threads = []
        threads.append(threading.Thread(target=Sensor().start))
        for i in range(self.go_length): # 8 = 6 + 6 - 4
            Move(int(real_Move_pos[i]), 100, 100, self.pre_omega[i ,:])

        start_time = time.time()
        threads[0].start()
        for i in range(60):
            pos_list, time_list = Move(int(real_Move_pos[i+self.go_length]), 100, 100, self.pre_omega[i+self.go_length ,:])
            current_time = time.time()
            self.pos_omega_his.append(pos_list)
            self.time_omega_his.append(time_list)
            self.time_list_for_refine.append(current_time - start_time)
            if self.time_list_for_refine[i] > self.sample_time-5:
                break
    
        Move(int(self.mid_pos), 100, 100, self.pre_omega[i ,:])
        time.sleep(1) 
        return self.time_list_for_refine, self.time_omega_his, self.pos_omega_his
      
    def efficiency_cal(self):
        F = np.zeros([len(self.time_window_list), 1])
        P = np.zeros([len(self.time_window_list), 3]) #P_need, P_use, yita      
        reward_his = []
        for i in range (len(self.time_window_list)): 
            real_F = sum(self.time_window_list[i][:, 3])
            P_need = -sum(self.time_window_list[i][:, 5])            
            F[i, 0] = real_F / len(self.time_window_list[i]) 
            P_use = F[i, 0]* 0.077 #v = 0.3 
            P[i, 0] = P_need / len(self.time_window_list[i])
            P[i, 1] = P_use
            P[i, 2] = (P[i, 1]/P[i, 0])  * self.correct_scale 
            if F[i, 0] >= 0.01173:
                reward = np.power(10, P[i, 2]) - 1 
            else:
                reward = 0               
            reward = np.clip(reward, -1, 1)
            reward_his.append(reward)
        return P, np.array(reward_his).reshape(-1, 1)
    
    def power_cut(self, time_delay):
        time.sleep(2)
        device.write("OUTPut CH1,OFF")
        time.sleep(time_delay-5)
        device.write("OUTPut CH1,ON")
        time.sleep(3)

    def step(self, whole_pre_decision, omega, measure):
        self.pre_omega = omega
        try:
            self.interaction_with_env(whole_pre_decision, omega)
            self.Data_refinement()
            self.power_cut(30)
            # time.sleep(15)
            time_window_list, cal_data = self.reward_cal()
            P, episode_rewards = self.efficiency_cal()
            error = False
        except:
            print('error')
            self.power_cut(120)
            Move(int(self.mid_pos), 100, 100, omega[0 ,:])
            episode_rewards = 0
            P = 0
            error = True
        return episode_rewards, error, P, self.If_sensor_change_list




