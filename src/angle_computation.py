import os
import pandas as pd
import numpy as np
import toml
from anipose.compute_angles import compute_angles
from config import SESSIONS, get_session_dir, get_3d_points_path, get_angles_path, get_combined_data_path

def compute_joint_angles(session):
    """Compute joint angles for a given session."""
    config_path = os.path.join(get_session_dir(session), f'{session}_config.toml')
    config = toml.load(config_path)
    
    # Load 3D points
    points_3d = np.load(get_3d_points_path(session))
    
    # Convert 3D points to a DataFrame
    body_parts = config['labeling']['bodyparts']
    df_3d = pd.DataFrame(points_3d.reshape(-1, len(body_parts) * 3),
                         columns=[f"{bp}_{axis}" for bp in body_parts for axis in ['x', 'y', 'z']])
    df_3d['fnum'] = range(1, len(df_3d) + 1)
    
    # Compute angles
    labels_path = os.path.join(get_session_dir(session), f'{session}_3d_points.csv')
    df_3d.to_csv(labels_path, index=False)
    angles_path = get_angles_path(session)
    compute_angles(config, labels_path, angles_path)
    
    print(f"Computed angles saved to {angles_path}")
    return df_3d, angles_path

def combine_3d_points_and_angles(session, df_3d, angles_path):
    """Combine 3D points and computed angles into a single DataFrame."""
    angles_df = pd.read_csv(angles_path)
    angles_of_interest = ['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle', "left_ankle_angle", "right_ankle_angle"]
    angles_df = angles_df[angles_of_interest]
    
    combined_df = pd.concat([df_3d, angles_df], axis=1)
    combined_path = get_combined_data_path(session)
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined 3D points and angles saved to {combined_path}")

def process_angles():
    """Process angles for all sessions."""
    for session in SESSIONS:
        print(f"Processing angles for {session}")
        df_3d, angles_path = compute_joint_angles(session)
        combine_3d_points_and_angles(session, df_3d, angles_path)

if __name__ == "__main__":
    process_angles()