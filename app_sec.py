import os
import time
import torch
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import streamlit as st
import torch.nn.functional as F
from typing import Optional, Union
from t5_utils import get_texts, clean_text, process_predictions
from torch.utils.data import DataLoader
from utils import process_file, show_extracted_sentences, download_model
from bert_utils import BertClassifier, text_preprocessing
from transformers import BertTokenizer, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, SequentialSampler


if os.path.exists('./models') is False:
    download_model()

time.sleep(5)

# BERT Initialzation Section
class_names = ['Not Req', 'Req']
model_bert = BertClassifier(len(class_names))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_bert.load_state_dict(torch.load("models/best_model_state.bin", map_location=device))
model_bert = model_bert.to(device)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# T5 Initialization Section
type_ = 't5-base'
checkpoint = 'models/t5'
model_t5 = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer_t5 = T5Tokenizer.from_pretrained(type_)


# Functions section

def set_header():
    st.title("**Security Requirements Extraction Tool**")
    st.markdown("Short desciption: So far **only** .pdf format is supported by the app. Big documents **are not recommended**.")
    st.markdown("Supported language: English")
    st.markdown("Used models: BERT(based) and T5")
    st.markdown("Used datasets: PURE, PROMISE")
    st.markdown("By VDO Tech Team. Special thanks to Vyacheslav Yastrebov.")
    
@st.cache(suppress_st_warning=True)  
def bert_predict(model, test_dataloader):
    model.eval()
    all_logits = []
    for batch in test_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    return probs

@st.cache(suppress_st_warning=True)
def preprocessing_for_bert(data, MAX_LEN=200):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer_bert.encode_plus(
            text=text_preprocessing(sent),  
            add_special_tokens=True,       
            max_length=MAX_LEN,                  
            pad_to_max_length=True,                   
            return_attention_mask=True,
            truncation = True)
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks

@st.cache(suppress_st_warning=True)
def t5_predict(test_dataloader, model):
    pred = []
    model.eval()
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)
        with torch.no_grad():
            pred_ids = model.generate(
                input_ids=b_src_input_ids, 
                attention_mask=b_src_attention_mask
            )
            pred_ids = pred_ids.cpu().numpy()
            for pred_id in pred_ids:
                pred_decoded = tokenizer_t5.decode(pred_id)
                pred.append(pred_decoded)
    return pred

class T5Dataset(Dataset):
    def __init__(self, df, indices):
        super(T5Dataset, self).__init__()
        df = df.iloc[indices]
        self.texts = get_texts(df)
        self.tokenizer = tokenizer_t5
        self.src_max_length = 200
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        src_tokenized = self.tokenizer.encode_plus(
            self.texts[index], 
            max_length=self.src_max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt')
        src_input_ids = src_tokenized['input_ids'].squeeze()
        src_attention_mask = src_tokenized['attention_mask'].squeeze()

        return {'src_input_ids': src_input_ids.long(),
            'src_attention_mask': src_attention_mask.long()}

@st.cache(suppress_st_warning=True)
def bert_step(model, tokenizer, dataset):
    batch_size = 4
    inputs, masks = preprocessing_for_bert(dataset['Text'])
    data = TensorDataset(inputs, masks)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    probs = bert_predict(model, dataloader)
    preds = probs[:, 1]
    pred_idx = list(np.where(preds >= 0.5))
    sents = dataset.loc[dataset.index[pred_idx]]
    return sents

@st.cache(suppress_st_warning=True)
def t5_step(dataframe):
    dataset_size = len(dataframe)
    indices = list(range(dataset_size))
    req_data = T5Dataset(dataframe, indices)
    test_dataloader = DataLoader(req_data, batch_size=4)
    pre_preds = t5_predict(test_dataloader, model_t5)
    preds = process_predictions(pre_preds, dataframe)
    return list(dataframe[dataframe['Label']=='True']['Text'])


def main():
    set_header()
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
    if uploaded_file is not None:
        requirements = process_file(uploaded_file.getvalue())
        sents_req = bert_step(model_bert, tokenizer_bert, requirements)
        sents_sec = t5_step(sents_req)
        
        show_extracted_sentences(sents_sec)

        
if __name__ == "__main__":
    main()
