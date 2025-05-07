import imageio
import os
import argparse

parser = argparse.ArgumentParser(description="Create a GIF from images.")
parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing images')
args = parser.parse_args()
directory = args.directory

# Directory where your images are stored
directory = f'{directory}/plots/'
# get all the files with .png extensin at prefix 
image_files = os.listdir(directory)
print(image_files)

print(f"Found {len(image_files)} images.")
# Output path for the GIF
output_gif_path = f'{directory}/gif.gif'
print(output_gif_path)
# Create the GIF
with imageio.get_writer(output_gif_path, mode='I', duration=0.5) as writer:
    for filename in image_files:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved to {output_gif_path}")
