import os
import json
from sleap import Labels, Video, LabeledFrame, Instance, Skeleton
from sleap.info.feature_suggestions import FeatureSuggestionPipeline
from config import SESSIONS, CAMERAS, get_video_path, get_annotation_dir

def create_annotation_project(session, camera):
    """Create an annotation project for a given session and camera."""
    video_path = get_video_path(session, camera)
    save_dir = get_annotation_dir(session, camera)
    
    # Create FeatureSuggestionPipeline
    pipeline = FeatureSuggestionPipeline(
        per_video=50,
        scale=0.25,
        sample_method="stride",
        feature_type="hog",
        brisk_threshold=10,
        n_components=10,
        n_clusters=10,
        per_cluster=5,
    )
    
    # Load video and run pipeline
    video = Video.from_filename(filename=video_path)
    pipeline.run_disk_stage([video])
    frame_data = pipeline.run_processing_state()
    
    # Create labeled frames
    skeleton = Skeleton()
    json_path = os.path.join(os.path.dirname(__file__), "mouse19_skeleton.json")
    with open(json_path, 'r') as f:
        skl_json = json.load(f)
    skeleton = skeleton.from_dict(skl_json)
    
    labeled_frames = []
    for item in frame_data.items:
        frame_idx = item.frame_idx
        instances = [Instance(skeleton=skeleton)]
        labeled_frame = LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        labeled_frames.append(labeled_frame)
    
    # Create and save Labels object
    labels = Labels(labeled_frames)
    slp_path = os.path.join(save_dir, f"{session}_{camera}.slp")
    labels.save(slp_path)
    print(f"Created annotation project: {slp_path}")

def prepare_annotations():
    """Prepare annotation projects for all sessions and cameras."""
    for session in SESSIONS:
        for camera in CAMERAS:
            print(f"Preparing annotation project for {session} - {camera}")
            create_annotation_project(session, camera)

if __name__ == "__main__":
    prepare_annotations()