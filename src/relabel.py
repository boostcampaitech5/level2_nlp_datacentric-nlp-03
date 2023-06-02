import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification
from tokenization_kobert import KoBertTokenizer
from train import BERTDataset, get_probs
from torch.utils.data import DataLoader

checkpoint_path = "/opt/level2_nlp_datacentric-nlp-03/output/06-01-11:50:51/checkpoint-3000"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to("cuda:0")
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
model.eval()

crawl_df = pd.read_csv("/opt/level2_nlp_datacentric-nlp-03/data/crawl_rest_v1.csv")
crawl_dataset = BERTDataset(crawl_df, tokenizer, 64)

crawl_probs = get_probs(model, crawl_dataset)

crawl_prob_df = pd.DataFrame()
crawl_prob_df["ID"] = crawl_df["ID"]
crawl_prob_df["text"] = crawl_df["text"]
crawl_prob_df["probs"] = crawl_probs

crawl_prob_df.to_csv("crawl_probs.csv", index=False)