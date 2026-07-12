"""Organize Oxford Flowers-102 into an ImageFolder tree.

Reads the official label file (imagelabels.mat) and split file (setid.mat),
then moves images from jpg/ into Flowers/{train,val,test}/class_NNN/.
"""

import scipy.io
import os
import shutil

MAT_LABEL_FILE = 'imagelabels.mat'
MAT_SETID_FILE = 'setid.mat'
SOURCE_DIR = 'jpg'          # extracted from 102flowers.tgz
TARGET_ROOT = 'Flowers'


def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Source directory '{SOURCE_DIR}' not found.")
        print("   Extract 102flowers.tgz first, or fix SOURCE_DIR.")
        return

    print("Reading .mat files...")
    try:
        labels_mat = scipy.io.loadmat(MAT_LABEL_FILE)
        labels = labels_mat['labels'][0]

        setid_mat = scipy.io.loadmat(MAT_SETID_FILE)
        train_ids = set(setid_mat['trnid'][0])
        val_ids   = set(setid_mat['valid'][0])
        test_ids  = set(setid_mat['tstid'][0])

    except KeyError as e:
        print(f"❌ Key {e} not found in .mat file")
        return
    except Exception as e:
        print(f"❌ Failed to read .mat files: {e}")
        return

    print(f"{'-'*30}")
    print(f"✅ Total labels: {len(labels)}")
    print(f"✅ Train IDs: {len(train_ids)} (expected 1020)")
    print(f"✅ Valid IDs: {len(val_ids)} (expected 1020)")
    print(f"✅ Test  IDs: {len(test_ids)} (expected 6149)")
    print(f"{'-'*30}")

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_ROOT, split), exist_ok=True)

    move_count = 0
    missing_count = 0

    print("🚀 Organizing files...")

    # image IDs are 1-based
    total_imgs = len(labels)

    for i in range(total_imgs):
        img_id = i + 1
        label = labels[i]
        filename = f"image_{img_id:05d}.jpg"
        src_path = os.path.join(SOURCE_DIR, filename)

        if img_id in train_ids:
            split_name = 'train'
        elif img_id in val_ids:
            split_name = 'val'
        elif img_id in test_ids:
            split_name = 'test'
        else:
            continue

        class_folder = f"class_{label:03d}"
        target_dir = os.path.join(TARGET_ROOT, split_name, class_folder)

        os.makedirs(target_dir, exist_ok=True)
        dst_path = os.path.join(target_dir, filename)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            move_count += 1
        else:
            missing_count += 1

        if img_id % 1000 == 0:
            print(f"   Processed {img_id} / {total_imgs} images...")

    print("=" * 30)
    print("🎉 Done!")
    print(f"Moved: {move_count} images")
    if missing_count > 0:
        print(f"⚠️ Missing sources: {missing_count} images (check SOURCE_DIR)")

    print(f"\nDataset is ready at: ./{TARGET_ROOT}/")
    print(f"   {TARGET_ROOT}/{{train,val,test}}/class_xxx/...")


if __name__ == "__main__":
    main()
