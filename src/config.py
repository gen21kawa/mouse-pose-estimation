import os
import toml

# Base directory containing all session folders
BASE_DIR = "E:/complete-projects"

def load_session_config(session_name):
    """Load session-specific configuration from config.toml file."""
    config_path = os.path.join(BASE_DIR, session_name, "config.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for session {session_name}")
    return toml.load(config_path)

def get_session_dir(session_name):
    return os.path.join(BASE_DIR, session_name)

def get_camera_dir(session_name, camera):
    return os.path.join(get_session_dir(session_name), camera)

def get_video_path(session_name, camera):
    config = load_session_config(session_name)
    return os.path.join(config['video_dir'], f"{session_name}_{camera}.{config['video_format']}")

def get_annotation_dir(session_name, camera):
    return os.path.join(BASE_DIR, f"{session_name}_annotations", f"{session_name}_{camera}_annotations")

def get_output_video_path(session_name, camera):
    config = load_session_config(session_name)
    return os.path.join(get_annotation_dir(session_name, camera), f"{session_name}_{camera}_annotations.{config['output_video_format']}")

def get_pose_estimation_path(session_name):
    return os.path.join(get_session_dir(session_name), f"{session_name}_pose_estimation.h5")

def get_3d_points_path(session_name):
    return os.path.join(get_session_dir(session_name), f"{session_name}_3d_points.npy")

def get_angles_path(session_name):
    return os.path.join(get_session_dir(session_name), f"{session_name}_angles.csv")

def get_combined_data_path(session_name):
    return os.path.join(get_session_dir(session_name), f"{session_name}_3dpts_angles.csv")

def get_plot_path(session_name):
    return os.path.join(get_session_dir(session_name), f"{session_name}_angles.pdf")