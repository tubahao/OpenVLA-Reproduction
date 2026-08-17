import tensorflow_datasets as tfds
b = tfds.builder("libero_spatial_no_noops:1.0.0", data_dir="/root/autodl-tmp/datasets/modified_libero_rlds")
print("TFDS_BUILDER_OK", b.info.splits["train"].num_examples)
