import argparse
from project_setup import create_project_structure
from video_processing import process_videos
from annotation_preparation import prepare_annotations
from pose_estimation import run_pose_estimation
from triangulation import run_triangulation
from angle_computation import process_angles
from visualization import create_visualizations

def run_pipeline(steps):
    """Run the complete pipeline or specified steps."""
    step_functions = {
        'setup': create_project_structure,
        'videos': process_videos,
        'annotations': prepare_annotations,
        'pose': run_pose_estimation,
        'triangulation': run_triangulation,
        'angles': process_angles,
        'visualize': create_visualizations
    }
    
    for step in steps:
        if step in step_functions:
            print(f"Running step: {step}")
            step_functions[step]()
        else:
            print(f"Unknown step: {step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Pose Estimation Pipeline for Mouse Behavior Analysis")
    parser.add_argument('--steps', nargs='+', default=['all'],
                        choices=['all', 'setup', 'videos', 'annotations', 'pose', 'triangulation', 'angles', 'visualize'],
                        help='Steps of the pipeline to run')
    
    args = parser.parse_args()
    
    if 'all' in args.steps:
        steps = ['setup', 'videos', 'annotations', 'pose', 'triangulation', 'angles', 'visualize']
    else:
        steps = args.steps
    
    run_pipeline(steps)