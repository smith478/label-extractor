import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load your dataset
data = pd.read_csv('radiology_reports.csv')

# Split the dataset into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['report_text'], data['diagnosis'], test_size=0.2
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the texts
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
