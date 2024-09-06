import matplotlib.pyplot as plt
import pandas as pd
from config import SESSIONS, get_angles_path, get_plot_path, PLOT_WIDTH, PLOT_HEIGHT

def plot_angles(session):
    """Plot joint angles for a given session."""
    angles_path = get_angles_path(session)
    df = pd.read_csv(angles_path)
    
    angles_of_interest = ['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle', "left_ankle_angle", "right_ankle_angle"]
    df = df[angles_of_interest]
    
    # Plot only a subset of frames (e.g., 400-500)
    df = df[400:500]
    
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    df.plot()
    plt.ylabel('Angle (degrees)')
    plt.xlabel('Frame')
    plt.legend(fontsize='x-small')
    plt.title(f'Joint Angles for Session {session}')
    
    plot_path = get_plot_path(session)
    plt.savefig(plot_path)
    plt.close()
    print(f"Angle plot saved to {plot_path}")

def create_visualizations():
    """Create visualizations for all sessions."""
    for session in SESSIONS:
        print(f"Creating visualization for {session}")
        plot_angles(session)

if __name__ == "__main__":
    create_visualizations()