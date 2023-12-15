import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns


from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import accelerate

import datasets
from datasets import load_dataset

import spacy
import nltk
# import contractions
from nltk.corpus import stopwords

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report


import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

PROJECT_PATH = 'model'
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility

RANDOM_SEED = 42

def set_reproducibility(seed: int):
    """
    Set the same random seed to different sources of randomness

    :param seed: random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_reproducibility(RANDOM_SEED)

## Dataset loading and preparation

hf_train = load_dataset('trec', split='train')
hf_test = load_dataset('trec', split='test')

print(hf_train[0])

def update_labels(example):
    fine_label = example['fine_label']
    if fine_label in [29, 30, 31]:
        new_label = 0
    elif fine_label == 16:
        new_label = 1
    elif fine_label == 28:
        new_label = 2
    elif fine_label in [32, 33, 36]:
        new_label = 3
    elif fine_label in [34, 35]:
        new_label = 4
    elif fine_label in [2, 3, 5, 9, 10, 14, 15, 22]:
        new_label = 5
    elif fine_label == 8:
        new_label = 6
    elif fine_label == 11:
        new_label = 7
    elif fine_label == 39:
        new_label = 8
    elif fine_label == 45:
        new_label = 9
    elif fine_label == 41:
        new_label = 10
    elif fine_label in [48, 49, 40]:
        new_label = 11
    elif fine_label == 42:
        new_label = 12
    else:
        new_label = 13  # For 'CARDINAL' and others
    return {'updated_label': new_label}

# def update_labels(example):
#     new_label = 3 if example['fine_label'] == 28 else (7 if example['fine_label'] in [29, 30, 31] else example['coarse_label'])
#     return {'updated_label': new_label}

new_features = hf_train.features.copy()
new_features["updated_label"] = datasets.Value('int32')

hf_train = hf_train.map(update_labels, features=new_features)
hf_test = hf_test.map(update_labels, features=new_features)


# Convert the dataframes into pandas dataframes

train_df = pd.DataFrame(hf_train)
test_df = pd.DataFrame(hf_test)

# Drop fine_label column which will not be used

train_df.drop(['fine_label'], axis=1, inplace=True)
test_df.drop(['fine_label'], axis=1, inplace=True)

train_df.drop(['coarse_label'], axis=1, inplace=True)
test_df.drop(['coarse_label'], axis=1, inplace=True)

print(train_df.head(20))

"""Since a validation set was not provided, we will split the training set into two: 80% will be used as the training set and the remaining 20% will serve as the validation set."""

# Split the training set into train and validation set
tmp = train_df.copy()

train_df = train_df.sample(frac=0.8,random_state=RANDOM_SEED)
val_df = tmp.drop(train_df.index)

val_counts = train_df['updated_label'].value_counts()
labels = val_counts.index


# data analysis
# frequencies = [val_counts[label] for label in labels]

# fig = plt.figure(figsize=(12,6))
# fig.suptitle("Distribution of classes in training set", fontsize=16)
# bar = plt.bar(labels, frequencies)
# plt.xlabel("Classes")
# plt.ylabel("Count")
# plt.show()




## Transformer Approach

train_df = train_df.assign(labels=train_df['updated_label']).drop('updated_label', axis=1)
val_df = val_df.assign(labels=val_df['updated_label']).drop('updated_label', axis=1)
test_df = test_df.assign(labels=test_df['updated_label']).drop('updated_label', axis=1)

print(test_df.head())

BATCH_SIZE = 16
SAVE_PATH = os.path.join(PROJECT_PATH, 'transformers')
TRAIN_PATH = os.path.join(SAVE_PATH, 'train_saves')

"""### Loading the model and the tokenizer

First of all, we need to create two functions that will help us load the pre-trained model and the tokenizer. The tokenizer is a crucial component in this approach as it will be used to encode the text data into numerical representations that the model can understand. The functions will allow us to easily load the necessary components and start making predictions.
"""

LABELS_LIST= [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

def create_model(petrained_model_name : str, tokenizer ):
  """
  Loads and configure a pre-trained model via HuggingFace transformers library.

  :param petrained_model_name: name of the model to load
  :param tokenizer: the tokenizer of the model to be loaded

  :return:
    - model: pretrained encoder-decoder transformer model

  """
  id2label = {}
  label2id = {}

  for idx, label in enumerate(LABELS_LIST):
      id2label[idx] = label
      label2id[label] = idx



  model = AutoModelForSequenceClassification.from_pretrained(petrained_model_name,
                    num_labels=len(LABELS_LIST),
                    id2label= id2label,
                    label2id =label2id).to(device)


  model.config.early_stopping = True
  model.config.cls_token_id = tokenizer.cls_token_id
  model.config.eos_token_id = tokenizer.sep_token_id
  model.config.pad_token_id = tokenizer.pad_token_id

  return model

"""### Tokenize the dataset

In order to tokenize the dataset, we need to define the functions that will be used to preprocess the text data and convert it into numerical representations that can be fed into the Transformer model. The first step in this process is to convert each document into a sequence of tokens, where each token is a unique word or symbol in the text.

Once the text has been tokenized, the next step is to convert the tokens into numerical representations that the Transformer model can understand. This is usually done by mapping each token to a unique integer index, and then representing the tokenized documents as sequences of integers.

Once the tokenized documents have been converted into numerical representations, they can be fed into the Transformer model for training and testing. The model will learn to identify patterns and relationships in the data, and will be able to predict the class labels for new, unseen documents.
"""

def process_data_to_model_inputs(batch, tokenizer):
  '''
  This function allows to tokenize a batch of data

  :param batch: the batch of data that will be tokenized
  :param tokenizer: the model tokenizer that will be used to tokenize the batch

  :return batch: batch of data tokenized
  '''
  INPUT_MAX_LENGTH = 128
  input = []
  for text in batch["text"]:
    input.append(text)

  inputs = tokenizer(input,
                     padding="max_length",
                     truncation="longest_first",
                     max_length=INPUT_MAX_LENGTH)


  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  return batch

def tokenize_dataset(df : pd.DataFrame, tokenizer):
  '''
  This function allows to tokenize an entire dataset

  :param df: the dataframe that will be tokenized
  :param tokenizer: the model tokenizer that will be used to tokenize the dataset

  :return df_tokenized: dataset tokenized
  '''

  df_tokenized = datasets.Dataset.from_pandas(df)
  old_column_names = (list(df_tokenized.features.keys()))
  old_column_names.remove('labels')

  df_tokenized = df_tokenized.map(
    process_data_to_model_inputs,
    fn_kwargs={ "tokenizer": tokenizer},
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=old_column_names
  )
  df_tokenized.set_format(type="torch", device = device, columns=["input_ids", "attention_mask", "labels"])
  return df_tokenized

"""### Compute the metrics and generate the predictions

To evaluate the performance of the Transformer-based approach, we define functions to compute various metrics such as accuracy and F1 score. These metrics will allow us to determine the effectiveness of the model in correctly classifying the documents.
"""

def compute_metrics(eval_pred):
  """
  The function allows to compute the metrics for accuracy and micro f1 score
  """
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)

  f1_avg = f1_score(y_true=labels, y_pred=predictions, average='micro')
  accuracy= accuracy_score(y_true=labels, y_pred=predictions)

  metrics = {'accuracy': accuracy,
             'f1':  f1_avg}

  return metrics


def generate_predictions_for_evaluation(batch, model):
  '''
  The function allows to generate the predictions for  a given batch of data
  for a given model
  '''
  outputs = model(batch['input_ids'],attention_mask=batch['attention_mask']).logits.tolist()

  pred_label = np.argmax(np.array(outputs), axis=1)

  batch["predictions"] = pred_label

  return batch


def compute_evaluation_metrics(model, df_test, verbose = True):
    """
    The function allows to compute the metrics for unseen data
    It handles the production of the predictions and the production of
    the classification report
    """

    model_results = df_test.map(generate_predictions_for_evaluation,
                             fn_kwargs={ "model": model},
                             batched=True,
                             batch_size=BATCH_SIZE,
                             remove_columns=['input_ids', 'attention_mask']
                             )

    predictions = model_results['predictions'].cpu().data.numpy()
    y_true = model_results['labels'].cpu().data.numpy()

    if verbose:
      print(classification_report(y_true,predictions, zero_division =1))

    metrics = classification_report(y_true,predictions, output_dict=True, zero_division =1)

    return metrics

def get_trainer(model,tokenizer, training_args, df_train, df_val):
  '''
  This function allows to get a Trainer object that will be used
  to train our models

  :param model: the model that will be trained
  :param tokenizer: the model tokenizer
  :param training_args: TrainingArguments object that contains the trainer settings
  :param df_train: the dataset used for the training phase
  :param df_val: the dataset used fot the validation phase

  :return Trainer object
  '''
  trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    tokenizer= tokenizer,
    train_dataset=df_train,
    eval_dataset=df_val,
  )
  return trainer

"""### First Model - DistilBert

The first model we will test is DistilBERT. This is a pre-trained transformer model based on the original BERT architecture. It is a distilled version of BERT that has fewer parameters and requires less computational power, making it more suitable for some deployment scenarios.

Distilled models are a type of compressed deep learning models which are smaller and faster than the original models, yet still maintain a high level of accuracy. They are created by distilling the knowledge from a larger, more complex model into a smaller, more computationally efficient one. This is done by training the smaller model to predict the outputs of the larger model. The distilled model has the benefits of being faster, making it more suitable for deployment in real-world applications with limited computational resources. Additionally, distilled models can also be fine-tuned on specific tasks, allowing them to achieve even better performance than the original models.
"""

MODEL_NAME = 'distilbert-base-cased'


tokenizer =  AutoTokenizer.from_pretrained(MODEL_NAME)
model = create_model(MODEL_NAME, tokenizer)

tokenized_train_df = tokenize_dataset(train_df,tokenizer)
tokenized_val_df = tokenize_dataset(val_df, tokenizer)

"""#### Training"""

training_args = TrainingArguments(
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    logging_steps=len(train_df) // BATCH_SIZE,
    output_dir=TRAIN_PATH,
    optim='adamw_torch',
    report_to='all',
)

trainer = get_trainer(
    model=model,
    tokenizer=tokenizer,
    training_args=training_args,
    df_train=tokenized_train_df,
    df_val=tokenized_val_df
)

print("SANITY CHECK")
trainer.evaluate()

print('TRAINING')
trainer.train(resume_from_checkpoint=False)
trainer.save_model(SAVE_PATH + "/" + MODEL_NAME)

del model
del tokenizer

"""#### Evaluation of the results"""

MODEL_PATH = SAVE_PATH + "/" + MODEL_NAME

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenized_test_df = tokenize_dataset(test_df, tokenizer)

report = compute_evaluation_metrics(model,tokenized_test_df)

print("The accuracy obtained with {} is: {}".format(MODEL_NAME,report['accuracy']))

"""The results obtained from the distilbert-base-cased model on the test set show that the model is able to predict the labels of the documents with a high degree of accuracy, with an accuracy of 0.956. This indicates that the model is able to make accurate predictions for the majority of the classes.

### Second Model - BERT base

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google. It is designed to handle natural language processing tasks such as sentiment analysis, text classification, and question answering.

The bidirectional nature of BERT makes it effective at capturing both the forward and backward context of a word in the sentence, providing more robust representations of the words. Due to its exceptional performance, BERT has become a popular choice for many NLP tasks, and has been fine-tuned on various tasks to obtain state-of-the-art results, including the dataset which is currently being explored
"""

MODEL_NAME = 'bert-base-cased'

tokenizer =  AutoTokenizer.from_pretrained(MODEL_NAME)
model = create_model(MODEL_NAME, tokenizer)

tokenized_train_df = tokenize_dataset(train_df,tokenizer)
tokenized_val_df = tokenize_dataset(val_df, tokenizer)

"""#### Training"""

training_args = TrainingArguments(
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # fp16=True,
    warmup_steps=500,
    logging_steps= len(train_df) // BATCH_SIZE,
    output_dir = TRAIN_PATH,
    optim='adamw_torch',
    report_to ='all',
)

trainer = get_trainer(model = model,
                    tokenizer = tokenizer,
                    training_args = training_args,
                    df_train = tokenized_train_df,
                    df_val = tokenized_val_df)

print("SANITY CHECK")
trainer.evaluate()

print('TRAINING')
trainer.train(resume_from_checkpoint = False)
trainer.save_model(SAVE_PATH + "/" + MODEL_NAME)

del model

"""#### Evaluation"""

MODEL_PATH = SAVE_PATH + "/" + MODEL_NAME

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenized_test_df = tokenize_dataset(test_df, tokenizer)



report = compute_evaluation_metrics(model,tokenized_test_df)

print("The accuracy obtained with {} is: {}".format(MODEL_NAME,report['accuracy']))


def generate_prediction(model, row):
  print(type(model))
  out = model(row['input_ids'], attention_mask=row['attention_mask'])


  pred = np.argmax(np.array(out), axis=1)

  return pred
