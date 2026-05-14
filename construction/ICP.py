import json, os
import numpy as np
import open3d as o3d
def get_largest_cluster(pcd, eps=0.02, min_points=10):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()
    
    if max_label < 0:
        print("未找到有效的连通域（全是噪声）")
        return None

    largest_cluster_id = -1
    max_count = 0

    for i in range(max_label + 1):
        count = np.sum(labels == i)
        if count > max_count:
            max_count = count
            largest_cluster_id = i
            
    print(f"找到 {max_label + 1} 个簇，最大簇包含 {max_count} 个点")

    indices = np.where(labels == largest_cluster_id)[0]
    return pcd.select_by_index(indices)

with open(os.getcwd()+ '/construction/data/frame_result/png_sequence.json','r',encoding='utf-8') as f:
    png_seq = json.load(f)

frame_data = np.load(os.getcwd()+ '/construction/data/frame_result/frame_point_result.npy',allow_pickle = True).item()
points_collection = frame_data['points_collection']
bcT_collection = frame_data['bcT_collection']


delete_frame_idx = []

####################### 单帧筛选 ##########################

min_points_num = 5000
for i in range (len(points_collection)):
    ## 如果某一帧分割失败、深度缺失严重，点数会明显偏少
    if np.asarray(points_collection[i]).shape[0] < min_points_num:
        delete_frame_idx.append(i)

    ## 对单帧点云做统计滤波，看被删掉的点占比。
    frame_points = np.asarray(points_collection[i])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame_points[:, :3])
    pcd_clean, ind = pcd.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=2.0
    )
    outlier_ratio = 1 - len(ind) / len(pcd.points)
    if outlier_ratio > 0.3:
        delete_frame_idx.append(i)

print('bad frames: ',delete_frame_idx)

points_collection_clean = [points_collection[i] for i in range(len(points_collection)) if i not in delete_frame_idx]
bcT_collection_clean = [bcT_collection[i] for i in range(len(bcT_collection)) if i not in delete_frame_idx]
png_seq_clean = [png_seq[i] for i in range(len(png_seq)) if i not in delete_frame_idx]
points_collection = points_collection_clean
bcT_collection = bcT_collection_clean
png_seq = png_seq_clean

########################### ICP配准与筛选 ##############################
threshold = 0.01
trans_init = np.eye(4)
shapes = [data.shape for data in points_collection]
reference_idx = np.argmax(shapes)

item = points_collection.pop(reference_idx)
points_collection.insert(0, item)

item = bcT_collection.pop(reference_idx)
bcT_collection.insert(0, item)

item = png_seq.pop(reference_idx)
png_seq.insert(0, item)

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(points_collection[0][:, :3])
target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
print(f'target png:{png_seq[0]}')
## 重新保存png_seq and frame_point_result
frame_data['bcT_collection'] = bcT_collection
frame_data['points_collection'] = points_collection
with open(os.getcwd()+ '/construction/data/frame_result/png_sequence.json','w',encoding='utf-8') as f:
    json.dump(png_seq,f)
np.save(os.getcwd()+ '/construction/data/frame_result/frame_point_result.npy', frame_data)


#for i in range (len(points_collection)):
#    if i+1 != len(points_collection):
#        source = o3d.geometry.PointCloud()
#        source.points = o3d.utility.Vector3dVector(points_collection[i+1][:, :3])
#        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
#        if i > 0 :
#            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
#        
#        reg_p2l = o3d.pipelines.registration.registration_icp(
#        source, target, threshold, trans_init,
#        o3d.pipelines.registration.TransformationEstimationPointToPlane())
#        #print(reg_p2l.fitness)
#        # 有效匹配点<70% 或 误差>xxx mm
#        if (reg_p2l.fitness < 0.7) or (reg_p2l.inlier_rmse > 0.003):
#            print(f"fitness:{reg_p2l.fitness}, rmse:{reg_p2l.inlier_rmse}")
#            print('bad quality of source image:', png_seq[i])
#            continue
#
#        # 平移过多，导致滑切 阈值设为5cm
#        if np.linalg.norm (reg_p2l.transformation[:3,3]) > 0.02: 
#            print(f'translation:{np.linalg.norm (reg_p2l.transformation[:3,3]) }, delete source:{png_seq[i] }')
#            continue
#
#        source.transform(reg_p2l.transformation)
#
#        target.paint_uniform_color([0.55, 0.72 , 0.36 ])  
#        # source.paint_uniform_color([0, 0, 1 ])  # 蓝色
#        # o3d.visualization.draw_geometries([target, source])
#
#
#        target = source + target
#        target = target.voxel_down_sample(voxel_size=0.001)
#        
#
#    else:
#        break
#
#target = get_largest_cluster(target)
#target, ind = target.remove_radius_outlier(
#    nb_points=12,
#    radius=0.004
#)
#o3d.visualization.draw_geometries([target])
#o3d.io.write_point_cloud(os.getcwd()+"/construction/data/frame_result/depression_target.pcd", target)


