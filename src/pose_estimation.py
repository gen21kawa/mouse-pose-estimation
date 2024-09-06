import os
import subprocess
from config import SESSIONS, CAMERAS, get_video_path, get_annotation_dir, SLEAP_MODEL_PATH

def run_sleap_train(session, camera):
    """Run SLEAP training for a given session and camera."""
    slp_path = os.path.join(get_annotation_dir(session, camera), f"{session}_{camera}.slp")
    output_path = os.path.join(get_annotation_dir(session, camera), f"{session}_{camera}_model")
    
    command = [
        "sleap-train",
        "single_instance.json",
        slp_path,
        "--output_prefix", output_path
    ]
    
    subprocess.run(command, check=True)
    print(f"Completed SLEAP training for {session} - {camera}")

def run_sleap_track(session, camera):
    """Run SLEAP tracking for a given session and camera."""
    video_path = get_video_path(session, camera)
    model_path = os.path.join(get_annotation_dir(session, camera), f"{session}_{camera}_model.json")
    output_path = os.path.join(get_annotation_dir(session, camera), f"{session}_{camera}_prediction.slp")
    
    command = [
        "sleap-track",
        video_path,
        "--model", model_path,
        "--tracking.tracker", "none",
        "-o", output_path,
        "--verbosity", "json",
        "--no-empty-frames"
    ]
    
    subprocess.run(command, check=True)
    print(f"Completed SLEAP tracking for {session} - {camera}")

def run_pose_estimation():
    """Run pose estimation for all sessions and cameras."""
    for session in SESSIONS:
        for camera in CAMERAS:
            print(f"Running pose estimation for {session} - {camera}")
            run_sleap_train(session, camera)
            run_sleap_track(session, camera)

if __name__ == "__main__":
    run_pose_estimation()