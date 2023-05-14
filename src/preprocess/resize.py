import os
from PIL import Image

# Set the folders to process
folders = ['boots', 'heels', 'sneakers']

# Set the target size and JPEG quality
target_size = (256, 256)
jpeg_quality = 85

# Process each folder
for folder in folders:
    # Get the full path of the folder
    folder_path = os.path.join('data', 'footwear', folder)

    # Create a new folder for resized images
    new_folder_path = folder_path + '_resized'
    os.makedirs(new_folder_path, exist_ok=True)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

    # Process each image file
    for filename in image_files:
        # Open the image and resize it
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as image:
            # Convert PNG to JPEG
            if image.format == 'PNG':
                image = image.convert('RGB')

            # Resize the image
            image = image.resize(target_size)

            # Save the resized image as JPEG with the specified quality
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            save_path = os.path.join(new_folder_path, new_filename)
            image.save(save_path, 'JPEG', quality=jpeg_quality)

            print(f'Resized and saved {filename} to {new_filename} in {new_folder_path}')
