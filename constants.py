import os

# TRAINING_APPLICATION_NAME = "training-script"
# PREDICTING_APPLICATION_NAME = "predicting-script"

MODEL_FOLDER = "./models"

# TMP_FOLDER_NAME = "tmp"
# TRAIN_DATASET_PATH = os.path.join(TMP_FOLDER_NAME, "train_data.pt")
# VALID_DATASET_PATH = os.path.join(TMP_FOLDER_NAME, "valid_data.pt")
# PREDICT_DATASET_PATH = os.path.join(TMP_FOLDER_NAME, "predict_data.pt")

# DEFAULT_EPOCHS = 20
# MAX_LENGTH = 100
MODEL_TYPE_T5 = "t5"
MODEL_FILENAME = f"{MODEL_TYPE_T5}"
MODEL_PATH_T5 = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

# SEC_LABEL = "sec"
# NONSEC_LABEL = "nonsec"
# OTHER_LABEL = "other"

# SEC_IDX = 1
# NON_SEC_IDX = 0

# PT_URL = "https://www.dropbox.com/s/2bf5gdlb90tdwsw/pytorch_model.bin?dl=1"
PT_T5_URL = "https://www.dropbox.com/s/99te51vrzi0oc8t/pytorch_model.bin?dl=1"
# CONFIG_URL = "https://www.dropbox.com/s/dpq4f238mhzegwh/config.json?dl=1"
CONFIG_T5_URL = "https://www.dropbox.com/s/if1zdohmbi3n4y3/config.json?dl=1"
PT_BERT_URL = "https://www.dropbox.com/s/7dr2vq5f5mi3stf/best_model_state.bin?dl=1"

PT_T5_PATH = os.path.join(MODEL_PATH_T5, "pytorch_model.bin")
CONFIG_T5_PATH = os.path.join(MODEL_PATH_T5, "config.json")
BERT_PATH = os.path.join(MODEL_FOLDER, "best_model_state.bin")