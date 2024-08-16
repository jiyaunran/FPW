# FPW(Frequency-domain Pixel-by-Pixel Watermarking)
This repo is the official code for https://openreview.net/forum?id=4hSqhUhF20&noteId=4hSqhUhF20

# Dependencies and Installation
- Python 3.9.18
- Pytroch 2.1.0+cu121
- Torchvision 0.16.0
- TensorboardX 2.6.2.2

# Training
Choose a proper noise ratio can maintain both steathiness and effectiveness.
For exam, we use 3 on CelebA(64x64) and Cifar10(32x32)
python train.py --data_dir PATH_TO_DATASET \
--image_resolution YOUR_IMAGE_RESOLUTION \
--noise_ratio 3 \
--output_dir "./model/" \
--batch_size 256 \
--num_epochs 30 \
--cuda 0

# Embedding
For clean_data_dir, it gives a choice of merging a clean dataset with watermarked one. If there is no need, please leave it the same as PATH_TO_DATASET.
poison_data_num give an option of poison image number. If all poison is wanted, leave a big number will do the work.
python embed_fingerprints.py --data_dir PATH_TO_DATASET \
--clean_data_dir PATH_TO_CLEAN_DATASET \
--batch_size 256 \
--noise_ratio 3 \
--image_resolution YOUR_IMAGE_RESOLUTION \
--poison_data_num AMOUT_OFFINGERPRINTED_IMAGES \
--output_dir OUTPUT_PATH

# Generate Masking
Use a clean dataset to constraint generation of mask.
python gen_proper_cor.py --constraint_data_dir PATH_TO_DATASET \
--data_dir PATH_TO_SUR_GEN_DATASET \
--batch_size 256 \
--image_resolution YOUR_IMAGE_RESOLUTION \
--decoder_path MODEL_PATH \
--output_path OUTPUT_PATH \
--cuda 0

# Detection
If the masking method is used, use --mask 1 and type in the MASK_PATH 
python detect_fingerprints.py --data_dir PATH_TO_DATASET \
--image_resolution 64 \
--decoder_path MODEL_PATH \
--batch_size 256 \
--thr DETECTION THR 1 \
--thr2 DETECTION THR 2 \
--mask 1 \
--mask_path MASK_PATH \
--cuda 0
