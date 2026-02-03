import os

def get_monai_data_dicts(images_dir, masks_dir):
    image_filenames = [img for img in os.listdir(images_dir) if img.endswith('.jpg')]
    data_dicts = []

    for img_name in image_filenames:
        base_name = img_name.replace('.jpg', '')
        
        ma_path = os.path.join(masks_dir, "1. Microaneurysms", base_name + "_MA.tif")
        he_path = os.path.join(masks_dir, "2. Haemorrhages", base_name + "_HE.tif")
        ex_path = os.path.join(masks_dir, "3. Hard Exudates", base_name + "_EX.tif")


        data_dicts.append({
            "image": os.path.join(images_dir, img_name),
            "ma": ma_path if os.path.exists(ma_path) else None,
            "he": he_path if os.path.exists(he_path) else None,
            "ex": ex_path if os.path.exists(ex_path) else None
        })
        
    return data_dicts