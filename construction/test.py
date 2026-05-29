import numpy as np
import open3d as o3d

unit_scale = 1000
depression_dir = "E:\HKUSTGZ\AAM\construction\data\completion_result\depression"
top_pcd = o3d.io.read_point_cloud(depression_dir+'/top_plane.pcd')
meta = np.load(depression_dir +'/meta.npz', allow_pickle=True)
grip_meta = np.load(depression_dir +'/gripper_meta.npz', allow_pickle=True)
orient_meta = np.load(depression_dir + '/orientation_meta.npz', allow_pickle=True)
grip_height_total = (grip_meta['grip_body_height'] + grip_meta['base_height'] + grip_meta['v_neck_height'])


#print(np.asarray(orient_meta['attach_center_oriented'] ) * unit_scale)

print( grip_height_total)
print(orient_meta['attach_center_oriented'])
print(orient_meta['full_box_z_height'])
print(meta['top_plane_center'])