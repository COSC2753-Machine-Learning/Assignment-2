import os
import random
import PIL.Image as Image
import imagehash
import pandas as pd
import matplotlib.pyplot as plt


def find_duplicates(df, base_dir, hash_size=8) -> pd.DataFrame:
    image_hashes = {}
    duplicates_count = 0
    base_dir = os.path.normpath(base_dir) + os.sep

    for index, row in df.iterrows():
        img_path = os.path.join(base_dir, row["ImgPath"])
        try:
            with Image.open(img_path) as img:
                # Convert image hash to a hex string immediately
                img_hash = str(imagehash.phash(img, hash_size))
                if img_hash in image_hashes:
                    image_hashes[img_hash].append(row["ImgPath"])
                    duplicates_count += 1
                else:
                    image_hashes[img_hash] = [row["ImgPath"]]
        except IOError as e:
            print(f"Error opening image {img_path}: {e}")

    # Identifying groups with more than one image
    duplicated_image_hashes = {hash_val: paths for hash_val, paths in image_hashes.items() if len(paths) > 1}
    duplicates_list = [(hash_val, path) for hash_val, paths in duplicated_image_hashes.items() for path in paths]
    duplicates_df = pd.DataFrame(duplicates_list, columns=['Hash', 'ImgPath'])

    return duplicates_df

def visualize_duplicates(duplicate_dataframe, num_samples) -> None:
    # Convert the DataFrame to a dictionary
    duplicates_image_hashes = duplicate_dataframe.groupby('Hash')['ImgPath'].apply(list).to_dict()

    # Select random k samples from the dictionary
    sample_indices = random.sample(list(duplicates_image_hashes.keys()), num_samples)

    fig, ax = plt.subplots(num_samples, 2, figsize=(16, 12))

    for iteration, sample_index in enumerate(sample_indices):
        paths = duplicates_image_hashes[sample_index]

        for i, path in enumerate(paths):
            if i >= 2:
                break

            im = Image.open(f'../Data/Furniture_Data/{path}')
            ax[iteration, i].imshow(im)
            if i > 0:
                ax[iteration, i].text(
                    0.5, 0.5, f'Duplication',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax[iteration, i].transAxes,
                    fontsize=14,
                    color='red',
                    weight='bold'
                )
            ax[iteration, i].set_title(path, fontsize=8)
            ax[iteration, i].axis('off')
    
    plt.subplots_adjust(wspace=-0.5, hspace=0.2)
    plt.show()