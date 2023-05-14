import os

from imagededup.methods import DHash


if __name__ == '__main__':

    folders = ('boots', 'heels', 'sneakers')

    for folder in folders:
        phasher = DHash()

        # Set the directory path where your images are located
        image_dir = os.path.join("data", "footwear", folder)

        # Calculate the image fingerprints
        encodings = phasher.encode_images(image_dir=image_dir)

        # Find the duplicates
        # duplicates = phasher.find_duplicates(encoding_map=encodings, scores=False)
        duplicates = phasher.find_duplicates_to_remove(encoding_map=encodings)

        print(duplicates)

        # Delete the duplicates
        for filename in duplicates:
            try:
                os.remove(os.path.join(image_dir, filename))
                print("Deleted duplicate:", filename)
            except FileNotFoundError:
                print("File not found:", filename)