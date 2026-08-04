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

def clear_error(piper, escape_joint=None, vel=10, clear_gripper=True):
    """
    不掉电清除 Piper 机械臂关节/夹爪错误。

    Parameters
    ----------
    piper : C_PiperInterface_V2
        已经 ConnectPort() 且已创建的 piper 对象。
    escape_joint : list[float] or tuple[float], optional
        清错后用于离开限位的安全关节角，单位 degree。
        例如关节限位卡住后可传入 [0, 170, -10, 0, 0, 0]。
        若为 None，则只清错、重新使能和切回 MOVE J 模式，不主动运动。
    vel : int
        恢复后的运动速度百分比，建议 5~10。
    clear_gripper : bool
        是否同时清除夹爪错误并使能夹爪。

    Notes
    -----
    这个函数不会调用 ResetPiper()，因此不会主动让机械臂掉电下垂。
    """
    print("========== CLEAR ERROR ==========")

    # 3. 清除所有关节错误码。joint_num=7 表示所有关节；clear_err=0xAE 表示清错
    for i in range(3):
        try:
            piper.JointConfig(7, 0x00, 0x00, 300, 0xAE)
            time.sleep(0.1)
        except Exception as e:
            print(f"JointConfig clear_err failed at try {i + 1}:", e)
            time.sleep(0.1)

    # 4. 清除夹爪错误并使能夹爪；0x03 表示 enable and clear error
    if clear_gripper:
        try:
            piper.GripperCtrl(0, 1000, 0x03, 0)
            time.sleep(0.1)
        except Exception as e:
            print("Gripper clear error failed:", e)

    # 5. 重新使能机械臂
    enabled = False
    if hasattr(piper, "EnablePiper"):
        for _ in range(50):
            try:
                if piper.EnablePiper():
                    enabled = True
                    break
            except Exception:
                pass
            time.sleep(0.02)
    if enabled:
        print("enable piper after clear: success")
    else:
        print("enable after clear: failed or unknown")

    # 6. 切回 CAN 控制 + MOVE J 模式，低速恢复
    try:
        if hasattr(piper, "MotionCtrl_2"):
            piper.MotionCtrl_2(0x01, 0x01, vel, 0x00)
        else:
            piper.ModeCtrl(0x01, 0x01, vel, 0x00)
        time.sleep(0.2)
    except Exception as e:
        print("set MOVE J mode failed:", e)

    return piper


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
    