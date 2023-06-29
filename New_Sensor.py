import os
import pandas as pd
import numpy as np
import time

class Sensor():
    def __init__(self):
        self.sensor_path = r"C:\Users\zhang\Desktop\Sensor\DataToExcel_background\bin\Debug\DataToExcel_background.exe"
        self.sensor_excel_path = r"C:\Users\zhang\Desktop\Experiment_Record"
        self.quick_sensor_path = r"C:\Users\zhang\Desktop\Quick_Sensor\DataToExcel_background\bin\Debug\DataToExcel_background.exe"
        self.long_sensor_path = r"C:\Users\zhang\Desktop\long_Sensor\DataToExcel_background\bin\Debug\DataToExcel_background.exe"

    def start(self):
        os.system(self.sensor_path)        # time.sleep(2.1)
        pass
    def quick_start(self):
        os.system(self.quick_sensor_path)        # time.sleep(2.1)
        pass
    def long_start(self):
        os.system(self.long_sensor_path)        # time.sleep(2.1)
        pass
       
    def sensor_data(self):
        self.excel_list = os.listdir(self.sensor_excel_path)
        self.name = self.excel_list[-1]
        if self.name[-7] == "_":  #给秒补零，来获得正确排序
            self.new_name = self.name[:-6] + "0" + self.name[-6:]
            os.rename(self.sensor_excel_path + "\\" + self.name, self.sensor_excel_path + "\\"
                      + self.new_name)
            self.excel_list = os.listdir(self.sensor_excel_path)
            self.name = self.excel_list[-1]
        
        if self.name[-10] == "_":  #给分补零，来获得正确排序
            self.new_name = self.name[:-9] + "0" + self.name[-9:]
            os.rename(self.sensor_excel_path + "\\" + self.name, self.sensor_excel_path + "\\"
                      + self.new_name)
            self.excel_list = os.listdir(self.sensor_excel_path)
            self.name = self.excel_list[-1]
        
        if self.name[-13] == "_":  #给时补零，来获得正确排序
            self.new_name = self.name[:-12] + "0" + self.name[-12:]
            os.rename(self.sensor_excel_path + "\\" + self.name, self.sensor_excel_path + "\\"
                      + self.new_name)
            self.excel_list = os.listdir(self.sensor_excel_path)
            self.name = self.excel_list[-1]
        self.excel_data = pd.read_excel(self.sensor_excel_path + "\\" + self.excel_list[-1])
        self.excel_data = np.array(self.excel_data)

        # reward = sum(self.excel_data)/len(self.excel_data)
        return self.excel_data
    
    def quick_sensor_check(self):
        self.quick_start() #5s
        time.sleep(4)
        excel_data = self.sensor_data()
        F_up_0 = sum(excel_data[:, 2])/excel_data.shape[0]
        F_down_0 = sum(excel_data[:, 1])/excel_data.shape[0]
        M_0 = sum(excel_data[:, 0])/excel_data.shape[0]
        return M_0, F_up_0, F_down_0