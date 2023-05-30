import os
import random
import numpy as np
import pandas as pd
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone, timedelta
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from tqdm import tqdm 
from utils import draw_eda
from sklearn.model_selection import train_test_split

from tokenization_kobert import KoBertTokenizer



class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, MAX_LENGTH):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=MAX_LENGTH)
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),  
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)

## Define Metric
f1 = evaluate.load('f1')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')

## Train Model
def train_(model,WANDB_NAME, SAVE_STEPS):
    training_args = TrainingArguments(
        report_to='wandb',                    # enable logging to W&B
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        logging_strategy='steps',
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_steps=100,
        eval_steps=SAVE_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        learning_rate= 2e-05,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon=1e-08,
        weight_decay=0.01,
        lr_scheduler_type='linear',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def evaluate_(model, tokenizer, dataset_val):
    ## Evaluate Model
    dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    model.eval()
    preds = []
    for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test),desc='TEST'):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    dataset_test['target'] = preds
    dataset_test.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)

    ## val_data
    preds = []
    for i, sample in tqdm(dataset_val.iterrows(), total=len(dataset_val), desc='VAL'):
        inputs = tokenizer(sample['text'], return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)
    dataset_val['preds'] = preds
    dataset_val.to_csv(os.path.join(OUTPUT_DIR, 'val_result.csv'), index=False)

    ## figure
    probs = draw_eda(val_df=dataset_val, test_df=dataset_test, OUTPUT_DIR=OUTPUT_DIR)
    wandb.config.update({'probs':probs})


if __name__=='__main__':
    ##### ARGUMENTS #####
    TRAIN_DIR = 'train.csv'
    MAX_LENGTH = 64
    SAVE_STEPS = 200
    ######################

    # wandb setting
    #os.environ['WANDB_DISABLED'] = 'true'
    WANDB_NAME = datetime.now(tz=timezone(timedelta(hours=9))).strftime("%m-%d-%H:%M:%S")
    wandb.init(project='Data-centric',
               entity='ggul_tiger',
               name= WANDB_NAME)
    wandb.config.update({'TRAIN_DIR': TRAIN_DIR})
    
    ## Set Hyperparameters
    SEED = 456
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output', WANDB_NAME)

    ## Load Tokenizer and Model
    model_name = 'monologg/kobert'
    tokenizer = KoBertTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)

    ## Define Dataset
    data = pd.read_csv(os.path.join(DATA_DIR, TRAIN_DIR))
    dataset_train, dataset_valid = train_test_split(data, test_size=0.3, random_state=SEED)

    data_train = BERTDataset(dataset_train, tokenizer, MAX_LENGTH)
    data_valid = BERTDataset(dataset_valid, tokenizer, MAX_LENGTH)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # train
    train_(model, WANDB_NAME, SAVE_STEPS)
    # evaluate
    evaluate_(model,tokenizer,dataset_valid)

