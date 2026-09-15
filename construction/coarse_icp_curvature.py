import argparse, json, os
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT=Path(os.getenv("AAM_PROJECT_ROOT",Path(__file__).resolve().parents[1]))
DATA_DIR=PROJECT_ROOT/"construction"/"data"; COARSE_DIR=DATA_DIR/"coarse_scan"
COARSE_POINT_FILE=COARSE_DIR/"coarse_point_result.npz"; COARSE_SEQ_FILE=COARSE_DIR/"coarse_png_sequence.json"; COARSE_SCANPOSE_FILE=COARSE_DIR/"coarse_scanpose.json"
OUTPUT_FUSED_PCD=COARSE_DIR/"coarse_fuse_curvature.pcd"; OUTPUT_RESULT_JSON=COARSE_DIR/"coarse_icp_curvature_result.json"; OUTPUT_TRANSFORM_NPZ=COARSE_DIR/"coarse_icp_curvature_transforms.npz"; DEBUG_DIR=COARSE_DIR/"curvature_debug"
VISUALIZE=False

TARGET_CUBE_NAME="front_high"; FALLBACK_TARGET_CUBE_NAME="left_high"; TARGET_MIN_POINTS=500; TARGET_MAX_POINTS=80000
GLOBAL_DIST=0.015; NORMAL_RADIUS=0.015; NORMAL_MAX_NN=30
CURV_K=30; CURV_PERCENTILE=85; CURV_SIM_THRESHOLD=0.85; CANDIDATE_K=10; REFINE_DIST=0.010; REFINE_MAX_ITER=50
FEATURE_WEIGHT=5.0; PLANE_WEIGHT=1.0; PLANE_SAMPLE_STEP=5
CONVERGE_TRANSLATION=1e-5; CONVERGE_ROTATION_DEG=0.01
MAX_FINAL_TRANSLATION_MM=35.0; MAX_FINAL_ROTATION_DEG=15.0; MAX_FINAL_RMSE_MM=5.0
OVERLAP_DISTANCE_THRESHOLD=0.010

def configure_paths(args):
    global DATA_DIR,COARSE_DIR,COARSE_POINT_FILE,COARSE_SEQ_FILE,COARSE_SCANPOSE_FILE,OUTPUT_FUSED_PCD,OUTPUT_RESULT_JSON,OUTPUT_TRANSFORM_NPZ,DEBUG_DIR,VISUALIZE
    if args.run_dir: DATA_DIR=Path(args.run_dir)/"construction"; COARSE_DIR=DATA_DIR/"coarse_scan"
    if args.coarse_scan_dir: COARSE_DIR=Path(args.coarse_scan_dir)
    COARSE_POINT_FILE=Path(args.coarse_point_file) if args.coarse_point_file else COARSE_DIR/"coarse_point_result.npz"
    COARSE_SEQ_FILE=Path(args.coarse_seq_file) if args.coarse_seq_file else COARSE_DIR/"coarse_png_sequence.json"
    COARSE_SCANPOSE_FILE=Path(args.coarse_scanpose_file) if args.coarse_scanpose_file else COARSE_DIR/"coarse_scanpose.json"
    OUTPUT_FUSED_PCD=Path(args.output_pcd) if args.output_pcd else COARSE_DIR/"coarse_fuse_curvature.pcd"
    OUTPUT_RESULT_JSON=Path(args.output_json) if args.output_json else COARSE_DIR/"coarse_icp_curvature_result.json"
    OUTPUT_TRANSFORM_NPZ=Path(args.output_transforms) if args.output_transforms else COARSE_DIR/"coarse_icp_curvature_transforms.npz"
    DEBUG_DIR=Path(args.debug_dir) if args.debug_dir else COARSE_DIR/"curvature_debug"
    VISUALIZE=bool(args.visualize); COARSE_DIR.mkdir(parents=True,exist_ok=True); DEBUG_DIR.mkdir(parents=True,exist_ok=True)

def load_json(path):
    if not Path(path).exists(): return []
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def save_json(path,data):
    with open(path,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)

def points_to_xyz(points):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2: points=points.reshape(-1,points.shape[-1])
    points=points[:,:3]
    return points[np.all(np.isfinite(points),axis=1)]

def load_points_from_npz(path):
    d=np.load(path,allow_pickle=True)
    if "all_points" in d and "offsets" in d:
        pts=np.asarray(d["all_points"],dtype=float); off=np.asarray(d["offsets"],dtype=int)
        return [points_to_xyz(pts[off[i]:off[i+1]]) for i in range(len(off)-1)]
    if "points_collection" not in d: raise KeyError(f"{path}: no point-cloud array found")
    return [points_to_xyz(x) for x in d["points_collection"].tolist()]

def load_dataset(point_file,seq_file,prefix="coarse"):
    plist=load_points_from_npz(point_file); seq=load_json(seq_file)
    if len(seq)!=len(plist):
        print(f"[WARNING] sequence length {len(seq)} != point-cloud length {len(plist)}")
        seq=[f"{prefix}_{i}" for i in range(len(plist))]
    return [{"index":i,"dataset":prefix,"name":Path(str(seq[i])).stem,"global_name":f"{prefix}:{Path(str(seq[i])).stem}","points":p} for i,p in enumerate(plist)]

def normalize_cube_name(name): return str(name).strip().lower().replace("-","_").replace(" ","_")

def load_cube_name_to_frame_name(path):
    out={}
    for r in load_json(path):
        if r.get("cube_name") is not None and r.get("idx") is not None: out[normalize_cube_name(r["cube_name"])]=f"coarse_scan_{int(r['idx'])+1}"
    return out

def select_target_idx(items,raw_counts,proc_counts):
    m=load_cube_name_to_frame_name(COARSE_SCANPOSE_FILE); name_to_idx={x["name"]:i for i,x in enumerate(items)}
    front=m.get(normalize_cube_name(TARGET_CUBE_NAME)); left=m.get(normalize_cube_name(FALLBACK_TARGET_CUBE_NAME)); fi=name_to_idx.get(front); li=name_to_idx.get(left)
    if fi is not None and TARGET_MIN_POINTS<=raw_counts[fi]<=TARGET_MAX_POINTS and proc_counts[fi]>0:
        return fi,{"rule":"front_high preferred","selected_cube_name":TARGET_CUBE_NAME,"selected_frame":front,"front_high_raw_points":int(raw_counts[fi]),"front_high_processed_points":int(proc_counts[fi])}
    if li is not None and proc_counts[li]>0:
        return li,{"rule":"front_high abnormal, fallback to left_high","selected_cube_name":FALLBACK_TARGET_CUBE_NAME,"selected_frame":left}
    valid=[i for i,n in enumerate(proc_counts) if n>0]
    if not valid: raise RuntimeError("No valid coarse cloud for target selection")
    ti=max(valid,key=lambda i:raw_counts[i])
    return ti,{"rule":"front_high/left_high missing or invalid, fallback to max raw points","selected_cube_name":None,"selected_frame":items[ti]["name"]}

def make_pcd(points):
    p=o3d.geometry.PointCloud(); p.points=o3d.utility.Vector3dVector(np.asarray(points,dtype=float)); return p

def copy_pcd(p):
    q=make_pcd(np.asarray(p.points).copy())
    if p.has_normals(): q.normals=o3d.utility.Vector3dVector(np.asarray(p.normals).copy())
    if p.has_colors(): q.colors=o3d.utility.Vector3dVector(np.asarray(p.colors).copy())
    return q

def fuse_clouds(a,b):
    out=copy_pcd(a); out+=copy_pcd(b); return out

def estimate_normals(p):
    if len(p.points)==0: return
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS,max_nn=NORMAL_MAX_NN)); p.normalize_normals()

def transform_angle(T):
    x=np.clip((np.trace(T[:3,:3])-1.0)/2.0,-1.0,1.0); return float(np.degrees(np.arccos(x)))

def frame_number(name): return str(name).split("_")[-1]

def compute_overlap_ratio_aligned(source,target,threshold=OVERLAP_DISTANCE_THRESHOLD):
    if len(source.points)==0 or len(target.points)==0: return 0.0
    tree=cKDTree(np.asarray(target.points)); d,_=tree.query(np.asarray(source.points),k=1,workers=-1); return float(np.mean(d<=threshold))

def global_icp(source,target):
    estimate_normals(source); estimate_normals(target)
    return o3d.pipelines.registration.registration_icp(source,target,GLOBAL_DIST,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPlane(),o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7,relative_rmse=1e-7,max_iteration=100))

def tangent_basis(n):
    ref=np.array([1.,0.,0.]) if abs(n[0])<0.9 else np.array([0.,1.,0.])
    e1=np.cross(n,ref); e1/=np.linalg.norm(e1)+1e-12; e2=np.cross(n,e1); e2/=np.linalg.norm(e2)+1e-12; return e1,e2

def curvature_descriptors(p):
    pts=np.asarray(p.points)
    if len(pts)==0: return np.zeros((0,4)),np.zeros(0),np.zeros((0,3))
    estimate_normals(p); normals=np.asarray(p.normals); k=min(CURV_K,len(pts)); tree=cKDTree(pts); _,idx=tree.query(pts,k=k,workers=-1)
    if k==1: idx=idx[:,None]
    desc=np.zeros((len(pts),4)); score=np.zeros(len(pts))
    for i in range(len(pts)):
        e1,e2=tangent_basis(normals[i]); d=pts[idx[i]]-pts[i]; u=d@e1; v=d@e2; w=d@normals[i]
        A=np.column_stack([u*u,u*v,v*v,u,v,np.ones(len(u))]); c=np.linalg.lstsq(A,w,rcond=None)[0]
        Hs=np.array([[2*c[0],c[1]],[c[1],2*c[2]]]); k1,k2=np.linalg.eigvalsh(Hs)
        if abs(k1)<abs(k2): k1,k2=k2,k1
        desc[i]=[k1*k2,0.5*(k1+k2),k1,k2]; score[i]=max(abs(k1),abs(k2))
    return desc,score,normals

def descriptor_similarity(s,t,flip=False):
    s=s.copy()
    if flip: s[1:]*=-1
    ns,nt=np.linalg.norm(s),np.linalg.norm(t)
    if ns<1e-10 and nt<1e-10: return 1.0
    if ns<1e-10 or nt<1e-10: return 0.0
    return float(np.clip(np.dot(s,t)/(ns*nt),0.0,1.0))


def build_correspondences(source_points,source_normals,source_desc,source_high,target_points,target_normals,target_desc,target_tree):
    distances,candidates=target_tree.query(source_points,k=min(CANDIDATE_K,len(target_points)),workers=-1)
    if candidates.ndim==1: candidates=candidates[:,None]; distances=distances[:,None]
    src_ids=[]; tgt_ids=[]; weights=[]; sims=[]
    for i in range(len(source_points)):
        if not source_high[i] and i%PLANE_SAMPLE_STEP!=0: continue
        for d,j in zip(distances[i],candidates[i]):
            if d>REFINE_DIST: break
            if source_high[i]:
                flip=np.dot(source_normals[i],target_normals[j])<0
                sim=descriptor_similarity(source_desc[i],target_desc[j],flip)
                if sim<CURV_SIM_THRESHOLD: continue
                w=FEATURE_WEIGHT
            else:
                sim=1.0; w=PLANE_WEIGHT
            src_ids.append(i); tgt_ids.append(j); weights.append(w); sims.append(sim); break
    return np.asarray(src_ids,int),np.asarray(tgt_ids,int),np.asarray(weights,float),np.asarray(sims,float)

def solve_weighted_point_to_plane(source_points,target_points,target_normals,weights):
    cross=np.cross(source_points,target_normals); A=np.column_stack([cross,target_normals]); b=-np.sum(target_normals*(source_points-target_points),axis=1)
    sw=np.sqrt(weights); x=np.linalg.lstsq(A*sw[:,None],b*sw,rcond=None)[0]
    delta=np.eye(4); delta[:3,:3]=R.from_rotvec(x[:3]).as_matrix(); delta[:3,3]=x[3:]; return delta

def curvature_similarity_icp(source_global,target):
    source_work=copy_pcd(source_global); target_work=copy_pcd(target)
    source_desc,source_score,_=curvature_descriptors(source_work); target_desc,target_score,target_normals=curvature_descriptors(target_work)
    if len(source_score)==0 or len(target_score)==0:
        return source_work,np.eye(4),np.zeros(len(source_score),bool),np.zeros(len(target_score),bool),0.0,0.0,[]
    source_threshold=np.percentile(source_score,CURV_PERCENTILE); target_threshold=np.percentile(target_score,CURV_PERCENTILE)
    source_high=source_score>=source_threshold; target_high=target_score>=target_threshold
    target_points=np.asarray(target_work.points); target_tree=cKDTree(target_points); T_total=np.eye(4); records=[]
    for iteration in range(REFINE_MAX_ITER):
        source_points=np.asarray(source_work.points); source_normals_now=np.asarray(source_work.normals)
        src_ids,tgt_ids,weights,sims=build_correspondences(source_points,source_normals_now,source_desc,source_high,target_points,target_normals,target_desc,target_tree)
        if len(src_ids)<6:
            print(f"iter={iteration+1:02d} insufficient correspondences: {len(src_ids)}"); break
        src=source_points[src_ids]; tgt=target_points[tgt_ids]; nrm=target_normals[tgt_ids]
        delta=solve_weighted_point_to_plane(src,tgt,nrm,weights); source_work.transform(delta); T_total=delta@T_total
        trans=np.linalg.norm(delta[:3,3]); rot_deg=np.degrees(R.from_matrix(delta[:3,:3]).magnitude())
        transformed_src=(delta[:3,:3]@src.T).T+delta[:3,3]; residual=np.abs(np.sum(nrm*(transformed_src-tgt),axis=1))
        records.append({"iteration":iteration+1,"correspondences":int(len(src_ids)),"feature_correspondences":int(np.sum(weights>PLANE_WEIGHT)),"plane_correspondences":int(np.sum(weights==PLANE_WEIGHT)),"mean_similarity":float(np.mean(sims)),"median_point_to_plane_mm":float(np.median(residual)*1000),"translation_mm":float(trans*1000),"rotation_deg":float(rot_deg)})
        if trans<CONVERGE_TRANSLATION and rot_deg<CONVERGE_ROTATION_DEG: break
    return source_work,T_total,source_high,target_high,source_threshold,target_threshold,records

def main():
    items=load_dataset(COARSE_POINT_FILE,COARSE_SEQ_FILE,"coarse")
    if not items: raise RuntimeError("No coarse point-cloud frames were loaded")
    clouds=[make_pcd(x["points"]) for x in items]; 
    raw_counts=[len(x["points"]) for x in items]; 
    proc_counts=[len(x.points) for x in clouds]
    target_idx,target_selection=select_target_idx(items,raw_counts,proc_counts)
    current_target=copy_pcd(clouds[target_idx]); 
    target_name=items[target_idx]["name"]; 
    target_num=frame_number(target_name)
    accepted=[target_name]; rejected=[]; 
    source_indices=[i for i in range(len(items)) if i!=target_idx]; 
    processing_order=[items[i]["name"] for i in source_indices]
    transforms=[np.eye(4) for _ in items]; 
    kept=np.zeros(len(items),bool); kept[target_idx]=True; records=[]
    o3d.io.write_point_cloud(str(DEBUG_DIR/f"scan_{target_num}_transformed.pcd"),current_target)
    print("========== Curvature coarse ICP target =========="); 
    print("target:",items[target_idx]["global_name"]); 
    print("target selection:",target_selection); 
    print("processing order:",processing_order)

    for step,source_idx in enumerate(source_indices,start=1):
        item=items[source_idx]; name=item["name"]; source=copy_pcd(clouds[source_idx]); target_points_before=len(current_target.points)
        print(f"\n========== Step {step}: {name} -> current fused target ==========")
        if len(source.points)==0:
            rejected.append(name); records.append({"step":step,"source_index":source_idx,"source":name,"kept":False,"reject_reasons":["empty cloud"]}); print("[REJECTED] empty cloud"); continue

        reg_global=global_icp(source,current_target); T_global=np.asarray(reg_global.transformation,float); source_global=copy_pcd(source); source_global.transform(T_global)
        global_translation_mm=float(np.linalg.norm(T_global[:3,3])*1000); global_rotation_deg=transform_angle(T_global)
        print(f"Global ICP: fitness={reg_global.fitness:.4f}, rmse={reg_global.inlier_rmse*1000:.3f} mm, translation={global_translation_mm:.3f} mm, rotation={global_rotation_deg:.3f} deg")

        source_final,delta_T,source_high,target_high,source_threshold,target_threshold,refine_records=curvature_similarity_icp(source_global,current_target)
        T_final=delta_T@T_global; final_translation_mm=float(np.linalg.norm(T_final[:3,3])*1000); final_rotation_deg=transform_angle(T_final)
        final_eval=o3d.pipelines.registration.evaluate_registration(source_final,current_target,GLOBAL_DIST,np.eye(4))
        final_rmse_mm=float(final_eval.inlier_rmse*1000); final_fitness=float(final_eval.fitness); final_overlap=compute_overlap_ratio_aligned(source_final,current_target)
        refinement_translation_mm=float(np.linalg.norm(delta_T[:3,3])*1000); refinement_rotation_deg=transform_angle(delta_T)

        keep=True; reasons=[]
        if final_translation_mm>MAX_FINAL_TRANSLATION_MM: keep=False; reasons.append(f"final translation {final_translation_mm:.3f} mm > {MAX_FINAL_TRANSLATION_MM:.1f} mm")
        if final_rotation_deg>MAX_FINAL_ROTATION_DEG: keep=False; reasons.append(f"final rotation {final_rotation_deg:.3f} deg > {MAX_FINAL_ROTATION_DEG:.1f} deg")
        if final_rmse_mm>MAX_FINAL_RMSE_MM: keep=False; reasons.append(f"final RMSE {final_rmse_mm:.3f} mm > {MAX_FINAL_RMSE_MM:.1f} mm")
        transforms[source_idx]=T_final; kept[source_idx]=keep

        n=frame_number(name); transformed_path=DEBUG_DIR/f"scan_{n}_transformed.pcd"; global_path=DEBUG_DIR/f"scan_{n}_after_global_icp.pcd"; source_high_path=DEBUG_DIR/f"scan_{n}_high_curvature.pcd"; target_high_path=DEBUG_DIR/f"target_before_scan_{n}_high_curvature.pcd"
        o3d.io.write_point_cloud(str(transformed_path),source_final); o3d.io.write_point_cloud(str(global_path),source_global)
        o3d.io.write_point_cloud(str(source_high_path),make_pcd(np.asarray(source_global.points)[source_high])); o3d.io.write_point_cloud(str(target_high_path),make_pcd(np.asarray(current_target.points)[target_high]))
        target_frames_before=accepted.copy()
        if keep:
            current_target=fuse_clouds(current_target,source_final); accepted.append(name)
            print(f"[ACCEPTED] translation={final_translation_mm:.3f} mm, rotation={final_rotation_deg:.3f} deg, RMSE={final_rmse_mm:.3f} mm, fitness={final_fitness:.4f}, overlap={final_overlap:.3f}")
        else:
            rejected.append(name); print("[REJECTED]","; ".join(reasons))

        records.append({"step":step,"source_index":source_idx,"source":name,"source_global_name":item["global_name"],"target_frames_before_registration":target_frames_before,"target_points_before_registration":target_points_before,"global_fitness":float(reg_global.fitness),"global_rmse_mm":float(reg_global.inlier_rmse*1000),"global_translation_mm":global_translation_mm,"global_rotation_deg":global_rotation_deg,"T_global":T_global.tolist(),"delta_T_curvature_similarity":delta_T.tolist(),"refinement_translation_mm":refinement_translation_mm,"refinement_rotation_deg":refinement_rotation_deg,"T_final":T_final.tolist(),"final_translation_mm":final_translation_mm,"final_rotation_deg":final_rotation_deg,"final_rmse_mm":final_rmse_mm,"final_fitness":final_fitness,"final_overlap_ratio":final_overlap,"source_curvature_threshold":float(source_threshold),"target_curvature_threshold":float(target_threshold),"refinement_iterations":refine_records,"transformed_cloud":str(transformed_path),"after_global_cloud":str(global_path),"source_high_curvature_cloud":str(source_high_path),"target_high_curvature_cloud":str(target_high_path),"kept":bool(keep),"reject_reasons":reasons,"target_points_after_registration":len(current_target.points)})

    o3d.io.write_point_cloud(str(OUTPUT_FUSED_PCD),current_target)
    np.savez(OUTPUT_TRANSFORM_NPZ,names=np.asarray([x["global_name"] for x in items],dtype=object),frame_names=np.asarray([x["name"] for x in items],dtype=object),transforms=np.asarray(transforms,float),kept=kept,target_index=np.asarray(target_idx,int))
    result={"mode":"sequential_cumulative_curvature_icp","coarse_point_file":str(COARSE_POINT_FILE),"coarse_seq_file":str(COARSE_SEQ_FILE),"coarse_scanpose_file":str(COARSE_SCANPOSE_FILE),"output_fused_pcd":str(OUTPUT_FUSED_PCD),"output_transforms_npz":str(OUTPUT_TRANSFORM_NPZ),"debug_dir":str(DEBUG_DIR),"target_index":target_idx,"target_name":target_name,"target_global_name":items[target_idx]["global_name"],"target_selection":target_selection,"processing_order":processing_order,"accepted_frames":accepted,"rejected_frames":rejected,"final_fused_points":len(current_target.points),"filter_rule":{"max_final_translation_mm":MAX_FINAL_TRANSLATION_MM,"max_final_rotation_deg":MAX_FINAL_ROTATION_DEG,"max_final_rmse_mm":MAX_FINAL_RMSE_MM,"overlap_distance_threshold_m":OVERLAP_DISTANCE_THRESHOLD,"note":"overlap is diagnostic only"},"algorithm_parameters":{"global_dist_m":GLOBAL_DIST,"normal_radius_m":NORMAL_RADIUS,"normal_max_nn":NORMAL_MAX_NN,"curv_k":CURV_K,"curv_percentile":CURV_PERCENTILE,"curv_similarity_threshold":CURV_SIM_THRESHOLD,"candidate_k":CANDIDATE_K,"refine_dist_m":REFINE_DIST,"refine_max_iter":REFINE_MAX_ITER,"feature_weight":FEATURE_WEIGHT,"plane_weight":PLANE_WEIGHT,"plane_sample_step":PLANE_SAMPLE_STEP},"records":records}
    save_json(OUTPUT_RESULT_JSON,result)
    print("\nCurvature coarse ICP finished"); print("Accepted:",accepted); print("Rejected:",rejected); print("Fused:",OUTPUT_FUSED_PCD); print("Result:",OUTPUT_RESULT_JSON); print("Transforms:",OUTPUT_TRANSFORM_NPZ); print("Debug:",DEBUG_DIR)
    if VISUALIZE:
        vis=copy_pcd(current_target); vis.paint_uniform_color([0.2,0.7,1.0]); o3d.visualization.draw_geometries([vis],window_name="Curvature coarse ICP fused point cloud")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,default=None); parser.add_argument("--coarse-scan-dir",type=Path,default=None); parser.add_argument("--coarse-point-file",type=Path,default=None); parser.add_argument("--coarse-seq-file",type=Path,default=None); parser.add_argument("--coarse-scanpose-file",type=Path,default=None)
    parser.add_argument("--output-pcd",type=Path,default=None); parser.add_argument("--output-json",type=Path,default=None); parser.add_argument("--output-transforms",type=Path,default=None); parser.add_argument("--debug-dir",type=Path,default=None); parser.add_argument("--visualize",action="store_true")
    args=parser.parse_args(); configure_paths(args); main()
