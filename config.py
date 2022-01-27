## dataset configuartion
image_channel = 3
batch_train = 16
batch_test = 1
out_level = "character"
image_dirs = [
                "../UIT-HWDB-dataset/UIT_HWDB_line/train_data",
                "../UIT-HWDB-dataset/UIT_HWDB_line/test_data",
                "../UIT-HWDB-dataset/UIT_HWDB_line_syn"
            ] # for making vocab
train_image_dirs = [
                        "../UIT-HWDB-dataset/UIT_HWDB_line/train_data",
                        "../UIT-HWDB-dataset/UIT_HWDB_line_syn"
                    ] # for training
test_image_dirs = [
                        "../UIT-HWDB-dataset/UIT_HWDB_line/test_data"
                    ] # for testing
image_size = (-1, 64)

## training configuration
max_epoch = 500
learning_rate = 1
checkpoint_path = "/content/gdrive/MyDrive/TransformerOCR/saved_models"
start_from = None

## model configuration
dropout = 0.5
d_model = 256