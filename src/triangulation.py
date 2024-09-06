import numpy as np
import os
import pandas as pd
import h5py
import toml
from anipose import triangulate
from config import SESSIONS, get_session_dir, get_pose_estimation_path, get_3d_points_path, CALIBRATION_BOARD_PATH

def load_2d_data(session):
    """Load 2D tracking data for a given session."""
    pose_estimation_path = get_pose_estimation_path(session)
    with h5py.File(pose_estimation_path, 'r') as f:
        tracks = np.array(f['tracks'])
    return tracks

def perform_triangulation(session):
    """Perform 3D triangulation for a given session."""
    config_path = os.path.join(get_session_dir(session), f'{session}_config.toml')
    config = toml.load(config_path)
    
    tracks_2d = load_2d_data(session)
    points_3d = triangulate(tracks_2d, config)
    
    output_path = get_3d_points_path(session)
    np.save(output_path, points_3d)
    print(f"Saved 3D points to {output_path}")

def run_triangulation():
    """Run 3D triangulation for all sessions."""
    for session in SESSIONS:
        print(f"Running 3D triangulation for {session}")
        perform_triangulation(session)

if __name__ == "__main__":
    run_triangulation()