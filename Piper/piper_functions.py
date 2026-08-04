import time
from piper_sdk import *

def disable(arm_name):
    piper = C_PiperInterface_V2(arm_name)
    piper.ConnectPort()
    while(piper.DisablePiper()):
        time.sleep(0.01)
        print("失能成功!!!!")
    time.sleep(2)
    return piper

def enable(arm_name):
    piper = C_PiperInterface_V2(arm_name)
    piper.ConnectPort()
    time.sleep(0.1)
    while( not piper.EnablePiper()):
        # print('enable failed!!!!')
        time.sleep(0.01)
    print("使能成功!!!!")
    return piper

def disconnect_port(arm_name):
    piper.DisconnectPort(thread_timeout=0.1)

def read_param(piper):
    piper.ArmParamEnquiryAndConfig(4)
    print(piper.GetGripperTeachingPendantParamFeedback())
    time.sleep(0.05)

    return piper

def go_zero(piper):
    piper.ModeCtrl(0x01, 0x01, 30, 0x00)
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    print('go zero')
    time.sleep(1)
    return piper

def move_to_pos(piper, X, Y, Z, RX, RY, RZ, vel = 20):  ## 外部输入单位是mm
    fac = 1000
    count = 0
    X = int(X* fac)
    Y = int(Y* fac)
    Z = int(Z* fac)
    RX = int(RX * fac)
    RY = int(RY * fac)
    RZ = int(RZ * fac)
    

    while True:
        if count == 0 :
            print('move to position')
            piper.MotionCtrl_2(0x01, 0x00, vel, 0x00)
            piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
            #piper.GripperCtrl(0,1000,0x02, 0)
            #piper.GripperCtrl(0,1000,0x01, 0)
            time.sleep(0.2)
            armStatus = piper.GetArmStatus()
            print(armStatus)
            time.sleep(1) 
            count += 1  
        else:
            break 

    return piper

def move_gripper(piper,range,effort = 1000):  ## unit 0.001mm, 外部输入单位是mm
    fac = 1000
    piper.GripperCtrl(0,1000,0x02, 0)
    piper.GripperCtrl(0,1000,0x01, 0)
    count = 0
    while True:
        if count == 0:
            print(piper.GetArmGripperMsgs())
            piper.GripperCtrl(abs(range*1000), effort, 0x01, 0)
            time.sleep(0.005)
            count +=1
        else: break
    
    return piper

def get_endpose(piper):
    print('get endpose info:  ')
    count = 0
    while True:
        if count == 0:
            time.sleep(0.1)
            count+=1
            enpose_info = piper.GetArmEndPoseMsgs()
        else: 
            break

    return enpose_info

def get_arm_status(piper):
    count = 0
    while True:
        if count == 0:
            time.sleep(0.1)
            arm_status = piper.GetArmStatus()
            time.sleep(0.1)
            count +=1
        else: break
    return arm_status

def move_joint(piper, joint1, joint2, joint3, joint4, joint5, joint6 , vel = 20):
    factor = 1000
    piper.GripperCtrl(0,1000,0x01, 0)
    count = 0
    while True:
        if count ==0:
            count  = count + 1
            joint_0 = round(joint1 * factor)
            joint_1 = round(joint2 *factor)
            joint_2 = round(joint3 *factor)
            joint_3 = round(joint4 *factor)
            joint_4 = round(joint5 *factor)
            joint_5 = round(joint6 *factor)

            piper.MotionCtrl_2(0x01, 0x01, vel, 0x00)
            piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            
            time.sleep(0.005)
            count +=1
        else: 
            print(piper.GetArmStatus())
            break
    
    return
     


# 测试代码
if __name__ == "__main__":
    #piper = disable('r_piper')
    piper = enable('r_piper')

    piper.GripperTeachingPendantParamConfig(100, 70, 1)
    piper.ArmParamEnquiryAndConfig(4)
    #piper = read_param(piper)
    #piper = go_zero(piper)
    #time.sleep(1)
    #piper = move_joint(piper, 0,30,-15,0,0,0)

    piper = move_joint(piper,       
      -2.54,
      87.73,
      -38.43,
      11.67,
      -4.24,
      15.33,
      vel = 10 )
    