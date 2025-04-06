import imageio
import os

# Directory where your images are stored
image_dir = './'
prefix = 'codebook_usage_distribution_'
ext = '.png'

# Gather all image file paths in order of step
image_files = []
step = 0
while True:
    filename = f'{prefix}{step}{ext}'
    path = os.path.join(image_dir, filename)
    if not os.path.exists(path):
        break  # Stop when the file doesn't exist
    image_files.append(path)
    step += 1

# Output path for the GIF
output_gif_path = os.path.join(image_dir, 'codebook_usage_distribution.gif')

# Create the GIF
with imageio.get_writer(output_gif_path, mode='I', duration=0.5) as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved to {output_gif_path}")
