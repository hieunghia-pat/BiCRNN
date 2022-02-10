## dataset configuartion
image_channel = 3
batch_train = 16
batch_test = 1
out_level = "character"
image_dirs = [
                "../UIT-HWDB-dataset/UIT_HWDB_line_v2/train_data",
                "../UIT-HWDB-dataset/UIT_HWDB_line_v2/test_data"
            ] # for making vocab
train_image_dirs = [
                        "../UIT-HWDB-dataset/UIT_HWDB_line_v2/train_data"
                    ] # for training
test_image_dirs = [
                        "../UIT-HWDB-dataset/UIT_HWDB_line_v2/test_data"
                    ] # for testing
image_size = (-1, 128)

## training configuration
max_epoch = 500
learning_rate = 1
checkpoint_path = "saved_models/UIT-HWDB-line-character-level"
start_from = None

## model configuration
dropout = 0.5
d_model = 256