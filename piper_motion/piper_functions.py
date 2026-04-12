import time
from piper_sdk import *

def enable(arm_name):
    piper = C_PiperInterface_V2(arm_name)
    piper.ConnectPort()
    time.sleep(0.1)
    while( not piper.EnablePiper()):
        print('enable failed!!!!')
        time.sleep(0.01)
    print("使能成功!!!!")
    return piper

def disable(arm_name):
    piper = C_PiperInterface_V2(arm_name)
    piper.ConnectPort()
    while(piper.DisablePiper()):
        time.sleep(0.01)
        print("失能成功!!!!")
    time.sleep(2)
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

def move_to_pos(piper, X, Y, Z, RX, RY, RZ):  ## 外部输入单位是mm
    fac = 1000
    count = 0
    X = int(X* fac)
    Y = int(Y* fac)
    Z = int(Z* fac)
    RX = int(RX* fac)
    RY = int(RY* fac)
    RZ = int(RZ* fac)
    

    while True:
        if count == 0 :
            print('move to position')
            piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
            piper.GripperCtrl(0, 1000, 0x01, 0)
            time.sleep(1) 
            count += 1  
        else:
            break 

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
     


# 测试代码
if __name__ == "__main__":
    #piper = disable('r_piper')
    piper = enable('r_piper')
    #piper = read_param(piper)
    #piper = go_zero(piper)
    #piper = move_to_pos(piper,-20,-80,120,0,85,-45 ) #feasible para: 30,-80,350,0,85,-45
    endpose_info = get_endpose(piper)
    print(endpose_info)
