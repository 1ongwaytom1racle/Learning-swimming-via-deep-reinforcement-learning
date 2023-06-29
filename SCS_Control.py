import os
import numpy as np
import time
if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
        
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

from scservo_sdk import *                    # Uses SCServo SDK library

class SCS_Control():
    def __init__(self):
        
        # Control table address
        self.ADDR_SCS_TORQUE_ENABLE     = 40
        self.ADDR_SCS_GOAL_ACC          = 41
        self.ADDR_SCS_GOAL_POSITION     = 42
        self.ADDR_SCS_GOAL_SPEED        = 46
        self.ADDR_SCS_PRESENT_POSITION  = 56
        
        # Default setting
        self.SCS_ID                      = 1                 # SCServo ID : 1
        self.BAUDRATE                    = 1000000           # SCServo default baudrate : 1000000
        # self.BAUDRATE                    = 115200           # SCServo default baudrate : 1000000
        self.DEVICENAME                  = 'COM3'    # Check which port is being used on your controller
                                                        # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
        
        self.SCS_MOVING_STATUS_THRESHOLD = 40          # SCServo moving status threshold
        # SCS_MOVING_SPEED            = 600         # SCServo moving speed
        # SCS_MOVING_ACC              = 100           # SCServo moving acc
        self.protocol_end                = 0           # SCServo bit end(STS/SMS=0, SCS=1)

        self.point = 0

        self.portHandler = PortHandler(self.DEVICENAME)
        
        # Initialize PacketHandler instance
        # Get methods and members of Protocol
        self.packetHandler = PacketHandler(self.protocol_end)     
        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch()
            quit()
        
        # Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch()
            quit()
        self.f = open("test.txt","w")
    def reset(self):
        self.point = 0
    
    def write_point(self):
        print(self.point)
        return self.point
    
    def Move(self,goal_position,SCS_MOVING_SPEED,SCS_MOVING_ACC, speed_array):
        self.portHandler.openPort()
        start_time = time.time()
        omega_list = []
        pos_list = []
        time_list = []
        done = False
        self.scs_goal_position = goal_position
        self.packetHandler.write1ByteTxRx(self.portHandler, self.SCS_ID, 
                                          self.ADDR_SCS_GOAL_ACC, SCS_MOVING_ACC)
        self.packetHandler.write2ByteTxRx(self.portHandler, self.SCS_ID, 
                                      self.ADDR_SCS_GOAL_SPEED, SCS_MOVING_SPEED)
        while 1:
            # Write SCServo goal position
            self.packetHandler.write2ByteTxRx(self.portHandler, self.SCS_ID, 
                                              self.ADDR_SCS_GOAL_POSITION, self.scs_goal_position)
            self.packetHandler.write2ByteTxRx(self.portHandler, self.SCS_ID, 
                                          self.ADDR_SCS_GOAL_SPEED, SCS_MOVING_SPEED)

            while 1:
                # Read SCServo present position
                self.scs_present_position_speed, scs_comm_result, scs_error = self.packetHandler.read4ByteTxRx(self.portHandler, self.SCS_ID, self.ADDR_SCS_PRESENT_POSITION)
                if scs_comm_result != COMM_SUCCESS:
                    print(self.packetHandler.getTxRxResult(scs_comm_result))
                elif scs_error != 0:
                    print(self.packetHandler.getRxPacketError(scs_error))
        
                self.scs_present_position = SCS_LOWORD(self.scs_present_position_speed)

                pos_list.append(self.scs_present_position)
                time_list.append(time.time() - start_time)
                self.scs_present_speed = SCS_HIWORD(self.scs_present_position_speed)
                # print("GoalPos:%03d PresPos:%03d PresSpd:%03d" 
                #       % (self.scs_goal_position, self.scs_present_position, SCS_TOHOST(self.scs_present_speed, 15)))
                speed_index = np.min([abs(pos_list[0] - self.scs_present_position), 499])
                SCS_MOVING_SPEED = speed_array[speed_index]    
                break
                # scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.SCS_ID, self.ADDR_SCS_GOAL_SPEED, SCS_MOVING_SPEED)

                # if not (abs(self.scs_goal_position - self.scs_present_position_speed) > self.SCS_MOVING_STATUS_THRESHOLD):
                #     break 
            # if time.time() - start_time > 3:
            #     self.scs_goal_position = self.scs_present_position
                # break
            if self.scs_goal_position == self.scs_present_position:
                break
        self.portHandler.closePort()        
        return pos_list, time_list
    
    def stop_shock(self,goal_position,SCS_MOVING_SPEED,SCS_MOVING_ACC):
        self.portHandler.openPort()
        scs_comm_result, scs_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.SCS_ID, self.ADDR_SCS_GOAL_ACC, SCS_MOVING_ACC)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(scs_error))
            
        scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.SCS_ID, self.ADDR_SCS_GOAL_SPEED, SCS_MOVING_SPEED)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(scs_error))
    
        while 1:
            # Write SCServo goal position
            scs_goal_position = goal_position
            scs_comm_result, scs_error = self.packetHandler.write2ByteTxRx(self.portHandler, self.SCS_ID, self.ADDR_SCS_GOAL_POSITION, scs_goal_position)
            if scs_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
            elif scs_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(scs_error))
            
            while 1:
                # Read SCServo present position
                self.scs_present_position_speed, scs_comm_result, scs_error = self.packetHandler.read4ByteTxRx(self.portHandler, self.SCS_ID, self.ADDR_SCS_PRESENT_POSITION)
                if scs_comm_result != COMM_SUCCESS:
                    print(self.packetHandler.getTxRxResult(scs_comm_result))
                elif scs_error != 0:
                    print(self.packetHandler.getRxPacketError(scs_error))
        
                self.scs_present_position = SCS_LOWORD(self.scs_present_position_speed)
                self.scs_present_speed = SCS_HIWORD(self.scs_present_position_speed)
                if not (abs(scs_goal_position - self.scs_present_position_speed) > self.SCS_MOVING_STATUS_THRESHOLD):
                    break 
                scs_goal_position = self.scs_present_position    
            if scs_goal_position == self.scs_present_position:
                  break
        self.portHandler.closePort()        

    def TORQUE_Check(self):
        
        scs_comm_result, scs_error = self.packetHandler.write1ByteTxRx(self.portHandler, SCS_ID, self.ADDR_SCS_TORQUE_ENABLE, 0)
        if scs_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(scs_comm_result))
        elif scs_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(scs_error))

