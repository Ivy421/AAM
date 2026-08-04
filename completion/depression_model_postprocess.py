"""
Compatibility wrapper for older depression gripper code.

The mesh postprocess utilities now live in mesh_generation.py, but
Depression_grip.py imports them from depression_model_postprocess.py.
"""

from mesh_generation import clean_trimesh, force_consistent_positive_volume

