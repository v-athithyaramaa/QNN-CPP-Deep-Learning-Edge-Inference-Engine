import os
import tarfile
import urllib.request
import pickle
import numpy as np
from PIL import Image

CALIB_DIR = "calibration_data"
CALIB_RAW_DIR = os.path.join(CALIB_DIR, "raw")
TEST_DIR = "test_data"
TEST_IMG_DIR = os.path.join(TEST_DIR, "images")

os.makedirs(CALIB_RAW_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_PATH = "cifar-10-python.tar.gz"

if not os.path.exists(TAR_PATH):
    print("[INFO] Downloading CIFAR-10 archive (~163 MB)...")
    urllib.request.urlretrieve(CIFAR_URL, TAR_PATH)
    print("[INFO] Download complete.")

def load_batch_from_tar(tar_obj, member_name):
    member = tar_obj.getmember(member_name)
    f = tar_obj.extractfile(member)
    data_dict = pickle.load(f, encoding="bytes")
    return data_dict[b"data"], data_dict[b"labels"]

with tarfile.open(TAR_PATH, "r:gz") as tar:
    members = {m.name: m for m in tar.getmembers()}
    b1_key = next(k for k in members if k.endswith("data_batch_1"))
    test_key = next(k for k in members if k.endswith("test_batch"))

    # 1. Calibration Data (500 samples: 50 per class) in NHWC layout
    print("[INFO] Generating 500 calibration .raw samples (50 per class) in NHWC layout...")
    images_b1, labels_b1 = load_batch_from_tar(tar, b1_key)

    calib_counts = {i: 0 for i in range(10)}
    input_list_entries = []

    for idx, (img_flat, label) in enumerate(zip(images_b1, labels_b1)):
        if calib_counts[label] < 50:
            calib_counts[label] += 1
            
            # CIFAR-10 is stored as 3072 uint8 values: R(1024), G(1024), B(1024)
            # Transpose (3, 32, 32) -> (32, 32, 3) NHWC to match QNN graph input
            img_hwc = img_flat.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.float32) / 255.0
            
            raw_filename = f"calib_{idx}_class_{label}.raw"
            raw_filepath = os.path.join(CALIB_RAW_DIR, raw_filename)
            img_hwc.tofile(raw_filepath)
            
            rel_path = raw_filepath.replace("\\", "/")
            input_list_entries.append(f"input:={rel_path}\n")
            
        if sum(calib_counts.values()) == 500:
            break

    input_list_path = os.path.join(CALIB_DIR, "input_list.txt")
    with open(input_list_path, "w") as f:
        f.writelines(input_list_entries)

    print(f"[SUCCESS] Wrote 500 calibration files to {CALIB_RAW_DIR}")
    print(f"[SUCCESS] Generated {input_list_path}")

    # 2. Test Images (50 PNGs) & Ground Truth CSV
    print("[INFO] Generating 50 PNG test images and ground_truth.csv...")
    images_test, labels_test = load_batch_from_tar(tar, test_key)

    test_counts = {i: 0 for i in range(10)}
    ground_truth_lines = ["image_name,label,class_name\n"]

    for idx, (img_flat, label) in enumerate(zip(images_test, labels_test)):
        if test_counts[label] < 5:
            test_counts[label] += 1
            
            img_hwc = img_flat.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
            img = Image.fromarray(img_hwc)
            
            png_name = f"test_{idx}_class_{label}.png"
            png_path = os.path.join(TEST_IMG_DIR, png_name)
            img.save(png_path)
            
            ground_truth_lines.append(f"{png_name},{label},{CIFAR10_CLASSES[label]}\n")
            
        if sum(test_counts.values()) == 50:
            break

    gt_csv_path = os.path.join(TEST_DIR, "ground_truth.csv")
    with open(gt_csv_path, "w") as f:
        f.writelines(ground_truth_lines)

    print(f"[SUCCESS] Saved 50 PNG test images to {TEST_IMG_DIR}")
    print(f"[SUCCESS] Generated {gt_csv_path}")