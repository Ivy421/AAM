import subprocess
import sys
import datetime as dt
timestamp = dt.now()

subprocess.run([sys.executable, "/home/smmg/AAM/1_defect_detection/realsense_capturing.py"])
subprocess.run([sys.executable, "/home/smmg/AAM/1_defect_detection/1_Rough_defect_detect.py"])
subprocess.run([sys.executable, "/home/smmg/AAM/1_defect_detection/2_Rough_defect_positioning.py"])
subprocess.run([sys.executable, "/home/smmg/AAM/1_defect_detection/3_reference_analysis.py"])
subprocess.run([sys.executable, "/home/smmg/AAM/1_defect_detection/4_multiview_solution.py"])
