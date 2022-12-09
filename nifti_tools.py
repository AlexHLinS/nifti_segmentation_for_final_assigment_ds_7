""" Created by Alex Shkil (C) 2022
    this module implements tools for operation with nifti files
    also conver dataset to COCO format
"""
import json

from PIL import Image
import nibabel
import numpy as np


def create_bitmap_mask(mask_data: np.array) -> Image.Image:
    """convert nifti mask data to b/w image object

    Args:
        mask_data (np.array): nifti mask data array

    Returns:
        Image.Image: b/w image object
    """
    bitmap = Image.new(mode='1', size=mask_data.shape, color=0)
    pixel_map = bitmap.load()
    for i in range(mask_data.shape[0]):
        for j in range(mask_data.shape[1]):
            pixel_map[i, j] = int(mask_data[i, j])
    return bitmap


def create_bitmap_slice(slice_data: np.array) -> Image.Image:
    """convert nifti data to bitmap image object

    Args:
        slice_data (np.array): nifti data array

    Returns:
        Image.Image: bitmap image object
    """
    bitmap = Image.new(mode='RGBA', size=slice_data.shape,
                       color=(255, 255, 255, 127))
    pixel_map = bitmap.load()
    for i in range(slice_data.shape[0]):
        for j in range(slice_data.shape[1]):
            pixel_map[i, j] = convert_nifti_to_4ch(int(slice_data[i, j]))
    return bitmap


def convert_nifti_to_4ch(nifti_value: int) -> tuple:
    """convert nifti point value to RGB with Alpha transparency

    Args:
        nifti_value (int): ifti point values array

    Returns:
        tuple: RGB with Alpha transparency
    """
    a = nifti_value // 1000
    b = (nifti_value - a * 1000) // 100
    c = (nifti_value - a*1000 - b*100)//10
    d = nifti_value % 10
    alpha = 255 if a else 127
    red = b * 255 // 10
    blue = c * 255 // 10
    green = d * 255 // 10

    return (red, blue, green, alpha)


def get_dataset_info(dataset_folder: str) -> dict:
    """returns information from dataset.json file

    Args:
        dataset_folder (str): path to dataset folder

    Returns:
        dict: the information loaded from dataset.json file
    """
    try:
        with open(dataset_folder+'dataset.json', 'r', encoding='UTF-8') as description_file:
            discription_raw = description_file.read()
    except FileNotFoundError:
        print('Dataset description file not found')

    description = json.loads(discription_raw)

    return description


def get_data_from_nifti_file(filename: str,
                             transponded: bool = False) -> np.ndarray:
    """Getting data from nifti file

    Args:
        filename (str): file name of the nifti file

    Returns:
        np.ndarray: data from nifti file
    """
    nifti_file = nibabel.load(filename=filename)
    nifti_file_data = nifti_file.get_fdata()
    if transponded:
        nifti_file_data = nifti_file_data.T
    return nifti_file_data


def get_list_of_shapes(description: dict) -> list:
    """_summary_

    Args:
        description (dict): _description_

    Returns:
        list: _description_
    """
    result = list()

    for traininf_source in description:
        print(get_data_from_nifti_file(str(traininf_source['image']).replace(
            './', 'datasets/Task02_Heart/')).shape)
    return result


def create_training_dataset_file(dataset_root_folder: str,
                                 data: np.array,
                                 labels: np.array,
                                 caption: str):
    """Create bitmap files for training dataset:
    dataset_root_folder/caption_image.bmp and dataset_root_folder/caption_mask.bmp 

    Args:
        dataset_root_folder (str): folser for storing training dataset
        data (np.array): X
        labels (np.array): y
        caption (str): caption for files
    """
    image = create_bitmap_slice(data)
    mask = create_bitmap_mask(labels)
    image.save(f'{dataset_root_folder}/{caption}_image.bmp')
    mask.save(f'{dataset_root_folder}/{caption}_mask.bmp')


def get_mask_segmentation_counts(mask: np.array,
                                 marker: int = 1) -> list:
    """generate a run-length-encoded (RLE) bit mask for marker value from numpy 2d mask array

    Args:
        mask (np.array): numpy 2d mask array
        marker (int, optional): marker value. Defaults to 1.

    Returns:
        list: run-length-encoded (RLE) bit mask

    Example:
    mack array:
     ([[0, 0, 0, 0],\n
       [0, 1, 1, 0],\n
       [0, 0, 1, 0],\n
       [0, 0, 0, 0]])
    - for marker equal to 0, result will be:
    [0, 5, 2, 3, 1, 5]
    - for marker equal to 1, result will be:
    [5, 2, 3, 1, 5]
    """
    result = list()
    mask_array = mask.copy()
    mask_array = mask_array.reshape(-1)

    counter = 0
    if mask_array[0] == marker:
        result.append(0)
    for index in range(1, mask_array.shape[0]):
        if mask_array[index] == mask_array[index-1]:
            counter += 1
        else:
            counter += 1
            result.append(counter)
            counter = 0
    result.append(counter+1)

    return result


def convert_nifis2bitmaps(description: dict, dest_folder: str = 'datasets/img',
                          relative_path: str = 'datasets/Task02_Heart/',
                          path_prefix_img: str = 'imagesTr/',
                          path_prefix_labels: str = '',
                          extension: str = '.nii.gz'):
    """_summary_

    Args:
        description (dict): _description_
        dest_folder (str, optional): _description_. Defaults to 'datasets/img'.
        relative_path (str, optional): _description_. Defaults to 'datasets/Task02_Heart/'.
        path_prefix_img (str, optional): _description_. Defaults to 'imagesTr/'.
        path_prefix_labels (str, optional): _description_. Defaults to ''.
        extension (str, optional): _description_. Defaults to '.nii.gz'.
    """
    for traininf_source in description:

        capt = str(traininf_source['image']).replace('./', '')\
                                            .replace(path_prefix_img, '')\
                                            .replace(extension, '')\
                                            .replace(path_prefix_labels, '')

        image_filename = str(traininf_source['image']).replace(
            './', relative_path)
        mask_filename = str(traininf_source['label']).replace(
            './', relative_path)
        transpored_data = get_data_from_nifti_file(
            filename=image_filename, transponded=True)
        transpored_mask = get_data_from_nifti_file(
            filename=mask_filename, transponded=True)

        layers_count = transpored_mask.shape[0]

        for layer in range(layers_count):
            caption = f'{capt}_layer{layer}'
            print(caption)
            create_training_dataset_file(dataset_root_folder=dest_folder,
                                         data=transpored_data[layer],
                                         labels=transpored_mask[layer],
                                         caption=caption)


def main() -> int:
    """Main entry point

    Returns:
        int: returns status code for OS
    """
    nifti_training_dataset_discription = get_dataset_info(
        'datasets/Task02_Heart/')['training']

    convert_nifis2bitmaps(description=nifti_training_dataset_discription)
    return 0


if __name__ == "__main__":
    main()
