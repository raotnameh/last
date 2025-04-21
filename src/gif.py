import imageio
import os
import argparse

parser = argparse.ArgumentParser(description="Create a GIF from images.")
parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing images')
args = parser.parse_args()
directory = args.directory

# Directory where your images are stored
prefix = f'{directory}/plots/codebook_usage_distribution_'
ext = '.png'

# Gather all image file paths in order of step
image_files = []
step = 0
while True:
    filename = f'{prefix}{step}{ext}'
    if not os.path.exists(filename):
        break  # Stop when the file doesn't exist
    image_files.append(filename)
    step += 1000


# Output path for the GIF
output_gif_path = f'{directory}/gif.gif'
print(output_gif_path)
# Create the GIF
with imageio.get_writer(output_gif_path, mode='I', duration=0.5) as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved to {output_gif_path}")
