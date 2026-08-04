import time
import sys, os, pathlib
sys.path.append('/home/smmg/AAM')
from piper_motion.piper_functions import *    
import numpy as np

pick_dir = '/home/smmg/AAM/pick_and_place/data'
endposes = np.load(pick_dir+'/pick_place_endpose.npz',allow_pickle=True)
pregrab_endpose = endposes['pregrab_endpose']
prefix_endpose = endposes['prefix_endpose']
grab_endpose = endposes['grab_endpose']
fix_endpose = endposes['fix_endpose']

print(prefix_endpose, fix_endpose)
piper_run = True   # 0
if piper_run == True:
    #piper = disable('r_piper')
    piper = enable('r_piper')

    piper.GripperTeachingPendantParamConfig(100, 70, 1)
    piper.ArmParamEnquiryAndConfig(4)
    #piper = read_param(piper)
    piper = go_zero(piper)
    time.sleep(2)
    piper = move_gripper(piper,50,1000)
    ###### grab object
    #piper = move_to_pos(piper, pregrab_endpose[0], pregrab_endpose[1], pregrab_endpose[2], pregrab_endpose[3],pregrab_endpose[4], pregrab_endpose[5]) 
    #time.sleep(5)
    #piper = move_to_pos(piper, grab_endpose[0], grab_endpose[1], grab_endpose[2]-2, grab_endpose[3], grab_endpose[4], grab_endpose[5], vel = 5)   ## 外部输入单位是mm,  degrees
    #time.sleep(3)
    #piper = move_gripper(piper,0,1000)
    #time.sleep(2)
    #piper = move_to_pos(piper, prefix_endpose[0], prefix_endpose[1], prefix_endpose[2], prefix_endpose[3], prefix_endpose[4], prefix_endpose[5] )
    #time.sleep(5)
    #piper = move_to_pos(piper, 350.97, 95.58, 244.06,175.26, -1.85, 130.19, vel =5)     
