import os
import cv2
import pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from wbia_miew_id.datasets import get_chip_from_img
from wbia_miew_id.etl import preprocess_data
from torchvision import transforms

def process_image(row, crop_bbox, preprocess_dir, chip_idx, target_size):
    image_path = row['file_path_orig']
    bbox = row['bbox']
    theta = row['theta'] if row['theta'] is not None else 0

    target_h, target_w = target_size

    image = cv2.imread(image_path)

    if crop_bbox:
        image = get_chip_from_img(image, bbox, theta)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image)
    # pil_image_resized = pil_image.resize((target_w, target_h))
    tensor_interpolate = transforms.Compose([transforms.ToTensor(), transforms.Resize((target_w, target_h), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToPILImage()])
    pil_image_resized = tensor_interpolate(pil_image)
    


    image_name = f"chip_{chip_idx:012}.jpg"
    out_path = os.path.join(preprocess_dir, image_name)
    pil_image_resized.save(out_path)
    return row.name, out_path


def preprocess_images(df, crop_bbox, preprocess_dir, target_size, num_workers=None):
    df['file_path_orig'] = df['file_path']

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, row, crop_bbox, preprocess_dir, chip_idx, target_size) for chip_idx, row in tqdm(df.iterrows(), total=len(df))]

    for future in as_completed(futures):
        index, out_path = future.result()
        df.at[index, 'file_path'] = out_path

    df['file_path'].apply(os.path.abspath)

    return df


def preprocess_dataset(config, preprocess_dir_images):

    preprocess_dir_train = os.path.join(preprocess_dir_images, 'train')
    preprocess_dir_val = os.path.join(preprocess_dir_images, 'val')
    preprocess_mapping_path = os.path.join(preprocess_dir_images, 'preprocess_mapping.csv')

    print("Preprocessing images. Destination: ", preprocess_dir_images)
    os.makedirs(preprocess_dir_train, exist_ok=True)
    os.makedirs(preprocess_dir_val, exist_ok=True)

    df_train_full = preprocess_data(config.data.train.anno_path, 
                            name_keys=config.data.name_keys,
                            convert_names_to_ids=True, 
                            viewpoint_list=config.data.viewpoint_list, 
                            n_filter_min=None, 
                            n_subsample_max=None,
                            use_full_image_path=config.data.use_full_image_path,
                            images_dir = config.data.images_dir,
                            )

    df_val_full = preprocess_data(config.data.val.anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=2, 
                                n_subsample_max=None,
                                use_full_image_path=config.data.use_full_image_path,
                                images_dir = config.data.images_dir
                                )


    target_size = (config.data.image_size[0],config.data.image_size[1])
    crop_bbox = config.data.crop_bbox

    df_train_full = preprocess_images(df_train_full, crop_bbox, preprocess_dir_train, target_size)
    df_val_full = preprocess_images(df_val_full, crop_bbox, preprocess_dir_val, target_size)

    df_preprocess_map = pd.concat([df_train_full, df_val_full])
    df_preprocess_map = df_preprocess_map[['image_uuid', 'file_path_orig', 'file_path']]
    print('Saving preprocess mapping to: ', preprocess_mapping_path)
    df_preprocess_map.to_csv(preprocess_mapping_path, index=False)

    
def load_preprocessed_mapping(df, preprocess_dir_images):
    preprocess_mapping_path = os.path.join(preprocess_dir_images, 'preprocess_mapping.csv')
    df_preprocess_map = pd.read_csv(preprocess_mapping_path)

    df = df.drop(columns=['file_path'])
    df = df.merge(df_preprocess_map, on='image_uuid', how='left')

    return df