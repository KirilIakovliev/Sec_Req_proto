import os

MODEL_FOLDER = "./models"
MODEL_TYPE_T5 = "t5"
MODEL_FILENAME = f"{MODEL_TYPE_T5}"
MODEL_PATH_T5 = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

PT_T5_URL = "https://www.dropbox.com/s/dc05grxthra0hth/pytorch_model.bin?dl=1"
CONFIG_T5_URL = "https://www.dropbox.com/s/ubcq8m53959lq0e/config.json?dl=1"
PT_BERT_URL = "https://www.dropbox.com/s/zatwjk97sc5tc5w/best_model_state.pth?dl=1"

PT_T5_PATH = os.path.join(MODEL_PATH_T5, "pytorch_model.bin")
CONFIG_T5_PATH = os.path.join(MODEL_PATH_T5, "config.json")
BERT_PATH = os.path.join(MODEL_FOLDER, "best_model_state.pth")
