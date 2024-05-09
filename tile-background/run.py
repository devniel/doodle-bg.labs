import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
import random
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte
import datetime

def create_stroke_mosaic(output_path, tiles_directory, tile_width, tile_height, bg_width, bg_height, bg_color=(254, 80, 0)):
    """
    Creates a mosaic image with a solid color background and black and white tiles merged such that
    white areas become transparent and black areas retain color, placed over it.
    
    Args:
    - output_path (str): Path to save the final mosaic image.
    - tiles_directory (str): Directory containing black and white tile images.
    - tile_width (int): Width of each tile.
    - tile_height (int): Height of each tile.
    - bg_width (int): Width of the background.
    - bg_height (int): Height of the background.
    - bg_color (tuple): Background color in RGB format. Default is orange (255, 165, 0).
    """
    # Create a solid color background image
    background = Image.new('RGB', (bg_width, bg_height), color=bg_color)

    # Define the dark orange color
    dark_orange = (171, 54, 0)  # A darker shade of orange

    # Load tiles and prepare masks
    tiles = []
    for filename in os.listdir(tiles_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            tile_path = os.path.join(tiles_directory, filename)
            original_tile = Image.open(tile_path).convert('L')  # Convert to grayscale
            resized_tile = original_tile.resize((int(tile_width // 1.2), int(tile_height // 1.2)), Image.LANCZOS)  # Resize to 50%
            mask = resized_tile.point(lambda p: p <= 128 and 255)  # Create a mask where white areas are masked out
            tile_image = ImageOps.colorize(resized_tile, dark_orange, dark_orange)  # Colorize the image to white

            # Convert PIL image to numpy array for denoising
            np_image = np.array(tile_image)
            float_image = img_as_float(np_image)  # Convert numpy array to float (needed for denoising)
            sigma_est = np.mean(estimate_sigma(float_image, channel_axis=-1))
            denoised_image = denoise_nl_means(float_image, h=2.0 * sigma_est, fast_mode=True,
                                              patch_size=7, patch_distance=11, channel_axis=-1)
            denoised_image = img_as_ubyte(denoised_image)  # Convert float image back to 8-bit bytes

            # Convert numpy array back to PIL image
            denoised_tile = Image.fromarray(denoised_image) 
            denoised_tile = denoised_tile.filter(ImageFilter.EDGE_ENHANCE_MORE)  # Enhance edges to reduce blurriness
  
            # Center the resized tile within the original tile dimensions
            centered_tile = Image.new('RGBA', (tile_width, tile_height), bg_color+(0,))
            centered_mask = Image.new('L', (tile_width, tile_height), 0)
            top_left_corner = ((tile_width - denoised_tile.width) // 2, (tile_height - denoised_tile.height) // 2)
            centered_tile.paste(denoised_tile, top_left_corner)
            centered_mask.paste(mask, top_left_corner)

            tiles.append((centered_tile, centered_mask))
    if not tiles:
        raise ValueError("No tile images found in the directory.")

    random.shuffle(tiles)

    # Place tiles on the background using masks
    num_tiles_x = int(np.ceil(bg_width / tile_width))
    num_tiles_y = int(np.ceil(bg_height / tile_height))
    
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            tile, mask = tiles[(x + y * num_tiles_x) % len(tiles)]
            background.paste(tile, (x * tile_width, y * tile_height), mask)

    # Save the final image
    background.save(output_path)

if __name__ == "__main__":
    create_stroke_mosaic(f'./output/mosaic-{datetime.datetime.now()}.jpg', './assets', 128, 128, 1024, 1024)