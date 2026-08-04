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
        print('enable failed!!!!')
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
            armStatus = piper.GetArmStatus()
            print(armStatus)
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

def move_joint(piper,position =  [0,0,0,0,0,0,0] ):
    factor = 1000
    piper.GripperCtrl(0,1000,0x01, 0)
    count = 0
    while True:
        if count ==0:
            count  = count + 1
            joint_0 = round(position[0]*factor)
            joint_1 = round(position[1]*factor)
            joint_2 = round(position[2]*factor)
            joint_3 = round(position[3]*factor)
            joint_4 = round(position[4]*factor)
            joint_5 = round(position[5]*factor)
            joint_6 = round(position[6]*1000*1000)
            # piper.ModeCtrl( 0x01,0x01,30, 0x00)
            piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
            piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
            print(piper.GetArmStatus())
            print(position)
            time.sleep(0.005)
        else: break
    
    return
     


# 测试代码
if __name__ == "__main__":
    #piper = disable('r_piper')
    piper = enable('r_piper')
    piper = move_to_pos(piper, 73.5  ,-6.24  ,264 ,128.6,66.9 ,121 )  ##俯视调整

    #piper = read_param(piper)
    #piper = go_zero(piper)

    
    #piper = move_to_pos(piper,27,3,265,144,73,139 ) #feasible para: 30,-80,350,0,85,-45
    #endpose_info = get_endpose(piper)
    #print(endpose_info)
    #print(piper.GetArmStatus())

    # j1 = (35726 + 60000)  / 1000    ## j1 控制机械臂在世界中左右旋转
    # j2 = (16677 + 45000) / 1000       ## j2 控制机械臂在世界中上下移动
    # j3 = (220335 + -10000)  / 1000    ## j3 控制机械臂在世界中上下移动
    # j4 = (146062 + -80000)  / 1000     ## j4 控制手腕绕BASE X 转动 （）
    # j5 = (60390 + 20000)  / 1000    # j5 控制手腕低头(正)
    # j6 = (159155 + 0000)   / 1000           # j6 控制手腕左右转（正右转）
    # time.sleep(1)
    # move_joint(piper, position = [j1, j2, j3, j4, j5, j6, 0])
# 
    # time.sleep(1)
    # print(piper.GetArmJointMsgs())
    # print(piper.GetArmStatus())

    '''
Joint 1:-8245
Joint 2:2506
Joint 3:-21565
Joint 4:6994
Joint 5:37937
Joint 6:3587

# 俯视
j2 += 40000
j3 -= 25000

#左侧视


#右侧视

    '''
