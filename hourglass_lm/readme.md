# Hourglass LLM

Developed on **Megatron Bridge**: [https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main)

---

### Container Environment Setup

1. Please pull and run following container on your environment

    ```
    nvcr.io/nvidia/nemo:25.11.01
    ```
2. Clone NVIDIA Megatron Bridge repo and switch to the following commit. Replace `Megatron-Bridge/` with the downloaded repo

    ```
    https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/b97eb72f5c7cdab32e196fd21afb90d2e3b97b9a#
    ```
3. Apply the patch to Megatron-Bridge: Update the downloaded repository by overlaying the custom patch files:

    ```bash
    cp -r Megatron-Bridge-patch/* Megatron-Bridge/
    ```

---

### Prepare Data

1. Please download all the files listed inside `preprocess/data.txt`. This might take a while.

2. Please save all the downloaded files under same directory. Those files are `.bin` files storing token ids of the training corpus.

3. Run following command to convert the data into Megatron format
    ```bash
    python3 preprocess/convert_bin_to_megatron.py \
        --input-dir BIN_DIR \
        --output-prefix OUTPUT_PREFIX
    ```
    - ``BIN_DIR``: the directory of the downloaded data
    - ``OUTPUT_PREFIX``: the prefix of the output files for Megatron format data. This script will produce two files: `OUTPUT_PREFIX.bin` and `OUTPUT_PREFIX.idx`.

---

### Pre-training Scripts

1. Edit `pretraining_script/run_pretraining.sh`. You should replace following variables
    - `megatron_bridge_home`: the absolute path to `Megatron-Bridge/`
    - `work_dir`: the absolute path to this directory

2. Properly prepare the training config, all the configs are under `pretraining_script/conf`. Specifically, you need to fill in following fields accordingly
    - `dataset.train_data_path`: The path to the generated data in the previous section
    - `checkpoint.save` and `checkpoint.load`: The path you want to use for saving and loading checkpoints
    - `logger.tensorboard_dir`: The path to store Tensorboard log

3.  **Run the following command to start pre-training**
    ```bash
    bash pretraining_script/run_pretraining.sh CONFIG_FILE
    ```
    > **Note:** Remember to update the Megatron Bridge path in this shell script.
    >
    > CONFIG_FILE: Megatron Bridge pre-training config file
