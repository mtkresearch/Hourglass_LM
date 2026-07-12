# Dataset Preparation

All datasets are plain `torchvision.datasets.ImageFolder` trees (one sub-directory per
class). By default the training script expects them under `<repo>/data/`; you can put
them anywhere and point the script there with `DATA_ROOT=/path/to/data`.

Expected final layout:

```
data/
├── car_data/       {train, test}/<class>/*.jpg          196 classes,  8,144 / 8,041
├── CIFAR-10/       {train, test}/<class>/*.jpg           10 classes, 50,000 / 10,000
├── CIFAR-100/      {train, test}/<class>/*.jpg          100 classes, 50,000 / 10,000
├── Flowers/        {train, val, test}/class_NNN/*.jpg   102 classes, 1,020 / 1,020 / 6,149
├── VOC-12/         {train, val}/<class>/*.jpg            20 classes, 13,609 / 13,841
├── ImageNet-100/   {train, val}/<class>/*.jpg           100 classes, 117,000 / 13,000
└── Places-30/      {train, val}/<class>/*.jpg            30 classes, 149,254 / 3,000
```

Run every command below from this `data_prep/` directory, with the target directory as
`../data/`:

```bash
mkdir -p ../data
```

---

## Stanford Cars (`cars`)

Download the "Stanford Car Dataset by classes folder" from Kaggle:
<https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder>

```bash
# after downloading the archive as CARS.zip
unzip CARS.zip -d ../data/
# the archive nests car_data/car_data/{train,test}; flatten one level:
cp -r ../data/car_data/car_data/train ../data/car_data/train
cp -r ../data/car_data/car_data/test  ../data/car_data/test
```

## CIFAR-10 (`cifar10`)

Already provided as an image-folder tree by <https://github.com/YoongiKim/CIFAR-10-images>:

```bash
wget https://github.com/YoongiKim/CIFAR-10-images/archive/refs/heads/master.zip -O CIFAR-10-images.zip
unzip -q CIFAR-10-images.zip
mv CIFAR-10-images-master ../data/CIFAR-10
```

## CIFAR-100 (`cifar100`)

Image-folder version from <https://github.com/cyizhuo/CIFAR-100-dataset>
(original data: <https://www.cs.toronto.edu/~kriz/cifar.html>):

```bash
wget https://github.com/cyizhuo/CIFAR-100-dataset/archive/refs/heads/main.zip -O CIFAR-100-dataset.zip
unzip -q CIFAR-100-dataset.zip
mv CIFAR-100-dataset-main ../data/CIFAR-100
```

## Oxford Flowers-102 (`flowers`)

Official source: <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>
Download `102flowers.tgz` (images), `imagelabels.mat` and `setid.mat` into this
directory, then:

```bash
tar -xzvf 102flowers.tgz          # produces jpg/
python3 organize_flowers.py       # builds Flowers/{train,val,test}/class_NNN/
mv Flowers ../data/
```

> Note: following common Flowers-102 fine-tuning practice, training uses the large
> official *test* split (6,149 images) and evaluation uses the *val* split (1,020
> images). `scripts/train.sh` already handles this.

## Pascal VOC-12 as classification (`voc12`)

Converts VOC2012 *detection* annotations into an object-crop *classification* dataset:
every non-difficult bounding box is cropped and saved under its class (20 classes).
The script downloads VOC2012 automatically via torchvision:

```bash
python3 organize_VOC-12.py \
  --voc-root ./VOC_raw \
  --output-root ../data/VOC-12 \
  --download --allow-existing
```

## ImageNet-100 (`imagenet100`)

Built from the HuggingFace dataset `ilee0022/ImageNet100` (100-class subset of
ImageNet-1k; 117,000 train / 13,000 val):

```bash
python3 organize_imagenet-100.py \
  --hf-dataset ilee0022/ImageNet100 \
  --output-root ../data/ImageNet-100 \
  --allow-existing
```

## Places-30 (`places30`)

A 30-scene subset of Places365 (small 256×256 variant). The script downloads
Places365 via torchvision (~25 GB) and copies the 30 selected classes
(149,254 train / 3,000 val). The class list is hardcoded in the script.

```bash
python3 organize_places-30.py \
  --output-root ../data/Places-30 \
  --no-classes-file --allow-existing \
  --download --mode copy
```
