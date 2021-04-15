import re
import streamlit as st

def clean_text(text):
    text = text.split()
    text = [x.strip() for x in text]
    text = [x.replace('\n', ' ').replace('\t', ' ') for x in text]
    text = ' '.join(text)
    text = re.sub('([.,!?()])', r' \1 ', text)
    return text

@st.cache()
def get_texts(df):
    texts = 'multilabel classification: ' + df['Text'].apply(clean_text)
    texts = texts.values.tolist()
    return texts

def process_predictions(predictions, dataframe):
    processed_labels = [re.sub("<pad>|</s>", "", label).strip() for label in predictions]
    dataframe["Label"] = processed_labels
    return dataframe
