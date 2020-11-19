#!pip install -q transformers
#!pip install -q s3fs

import warnings
warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm
import transformers
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import s3fs
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
import argparse
import sys


def f1_score_func(preds, labels):
    """
    function to calculate F-1 score for the given predicted labels and true labels
    :param preds: predicted values from the BERT model
    :param labels: true labels
    :return: the corresponding F-1 score
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def evaluate(dataloader_test, model, device):
    """
    function to evaluate the BERT model
    :param dataloader_test: test set
    :param model: the BERT model object
    :param device: the device that BERT model is working on
    :return: loss_val_avg, predictions, true_vals
    """
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in dataloader_test:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total / len(dataloader_test)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


def run_evaluation(dataloader_test, model, device, encoder, epoch, model_name):
    """
    function to run the evaluation for BERT model
    :param dataloader_test: test set
    :param model: the BERT model object
    :param device: which device to use
    :param encoder: the encoder used to covert raw labels
    :param epoch: the current epoch of the training
    :param model_name: name of the model
    :return: None
    """
    # Validation Loss and Validation F-1 Score
    val_loss, predictions, true_vals = evaluate(dataloader_test, model, device)
    val_f1 = f1_score_func(predictions, true_vals)
    print('Val Loss = ', val_loss)
    print('Val F1 = ', val_f1)
    # Validation Accuracy
    encoded_classes = encoder.classes_
    predicted_category = [encoded_classes[np.argmax(x)] for x in predictions]
    true_category = [encoded_classes[x] for x in true_vals]
    # accuracy score
    print('Accuracy Score = ', accuracy_score(true_category, predicted_category))
    # Classification Report
    reprot = classification_report(true_category, predicted_category)
    print(reprot)
    # Confusion Matrix
    confusion = confusion_matrix(true_category, predicted_category)
    confusion_df = pd.DataFrame(confusion, index=[1,2,3,4,5], columns=[1,2,3,4,5])
    print(confusion_df)
    plt.clf()
    sns.heatmap(confusion_df/np.sum(confusion_df), annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(model_name+'-'+str(epoch)+'.png')


def check_gpu():
    """
    function to check whether we are using GPU or not
    :return: None
    """
    # Get the GPU device name.
    device_name = tf.test.gpu_device_name()
    # The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')


def main(num_epoch, max_length, batch_size, model_name):
    """
    main function to train and evaluate BERT model
    :param num_epoch: number of epochs for training the model
    :param max_length: max length of the input string for training
    :param batch_size: batch size for training the model
    :param model_name: the name of the BERT model
    :return: None
    """
    # check whether uses gpu or not
    check_gpu()
    # print model info
    print('Start training BERT model.')
    print('Number of epochs: ', num_epoch)
    print('Max input length: ', max_length)
    print('Batch size: ', batch_size)
    # read in data
    df = pd.read_csv("s3://msia490project/processed_video_reviews.csv").head(500000)
    df['reviewText'] = df['reviewText'].astype(str)
    df.head()
    # Encode the classes for BERT. We'll keep using the 3 labels we made earlier.
    encoder = LabelEncoder()
    df['score'] = encoder.fit_transform(df['score'])
    # Set X and y.
    X = df['reviewText']
    y = df['score']
    # Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Encoding the words in the training data into vectors.
    max_length = int(max_length)
    encoded_data_train = tokenizer.batch_encode_plus(
        X_train,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='pt'
    )
    # Encoding the words in the test data into vectors.
    encoded_data_test = tokenizer.batch_encode_plus(
        X_test,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='pt'
    )
    # Get inputs and attention masks from previously encoded data.
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(y_train.values)
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(y_test.values)
    # Instantiate TensorDataset
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    # Initialize the model.
    model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                       num_labels=5,
                                                                       output_attentions=False,
                                                                       output_hidden_states=False)
    # DataLoaders for running the model
    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=int(batch_size))
    dataloader_test = DataLoader(dataset_test,
                                 sampler=SequentialSampler(dataset_test),
                                 batch_size=int(batch_size))
    # Setting hyper-parameters
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    epochs = int(num_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)
    seed_val = 15
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda')
    # train the model
    model.to(device)
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[2].to(device),
                      }
            outputs = model(**inputs)
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
        # progress bar
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        # evaluate the model
        run_evaluation(dataloader_test, model, device, encoder, epoch, model_name)
    # save the model for future use/retrain
    torch.save({
        'epoch': num_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_name + '.tar')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", help="number of epochs for training the model")
    parser.add_argument("--batch_size", help="batch_size for training the model")
    parser.add_argument("--max_length", help="max length of reviews")
    parser.add_argument("--model_name", help="name of the bert model")
    args = parser.parse_args()
    model_name = 'BERT-max_length' + args.max_length + 'batch_size' + args.batch_size + 'num_epoch' + args.num_epoch
    sys.stdout = open(model_name + '.txt', 'w')
    main(args.num_epoch, args.max_length, args.batch_size, model_name)
