import imageio.v2 as imageio
import os
import argparse
import numpy as np
from PIL import Image



# python gif.py -d /raid/home/rajivratn/hemant_rajivratn/last/src/logs_skip_non_speech_true


# Set up argument parser
parser = argparse.ArgumentParser(description="Create an MP4 video from images.")
parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing images')
parser.add_argument('--fps', type=int, default=10, help='Frames per second for the video (default: 10)')
parser.add_argument('--num_samples', type=int, default=100, help='Number of images to sample (default: 100)')

args = parser.parse_args()
d = args.directory
fps = args.fps
num_samples = args.num_samples

# src/logs_skip_non_speech_true/plots/codebook_usage_distribution_10.png
# Directory where images are stored
directory = os.path.join(d, 'plots')
image_files = sorted(
    [f for f in os.listdir(directory) if f.endswith('.png')],
    key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
)

# Select up to `num_samples` images evenly spaced
n = len(image_files)
if n <= num_samples:
    selected_files = image_files
else:
    indices = np.linspace(0, n - 1, num=num_samples, dtype=int)
    selected_files = [image_files[i] for i in indices]

# Full paths of selected images
image_paths = [os.path.join(directory, f) for f in selected_files]

# Read the first image to determine the target size
first_image = Image.open(image_paths[0])
width, height = first_image.size  # Get the width and height of the first image

# Output MP4 path
output_video_path = os.path.join(d, 'video.mp4')
print(f"Saving MP4 to {output_video_path}")

# Write MP4 with ffmpeg explicitly
with imageio.get_writer(output_video_path, fps=fps, codec='libx264', format='ffmpeg') as writer:
    for path in image_paths:
        frame = Image.open(path)

        # Resize the frame to match the first image size
        resized_frame = frame.resize((width, height))

        # Convert the resized frame to numpy array and append to video
        writer.append_data(np.array(resized_frame))

print("Video saved.")
