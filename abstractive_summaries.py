#!/usr/bin/env python3

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_BASKET = [
    # "google/pegasus-aeslc",
    # "google/pegasus-arxiv",
    # "google/pegasus-big_patent",
    "google/pegasus-billsum",
    "google/pegasus-cnn_dailymail",
    # "google/pegasus-gigaword",
    # "google/pegasus-large",
    "google/pegasus-multi_news",
    "google/pegasus-newsroom",
    # "google/pegasus-pubmed",
    # "google/pegasus-reddit_tifu",
    # "google/pegasus-wikihow",
    # "google/pegasus-xsum"
]
CUSTOM_MODEL_NAME = 'arawat/pegasus-custom-xsum'


def generate_summary(text, model_name):
    # print('torch.cuda.is_available() -> ' + str(torch.cuda.is_available()))
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    batch = tokenizer.prepare_seq2seq_batch(text, truncation=True, padding="longest", return_tensors="pt").to(
        torch_device
    )
    translated = model.generate(**batch)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]


def generate_summary_custom_model(text, model_name):
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CUSTOM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    batch = tokenizer.prepare_seq2seq_batch(text, truncation=True, padding="longest", return_tensors="pt").to(
        torch_device
    )
    translated = model.generate(**batch)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]


def generate_summaries(document):
    summary_basket = {}

    for model_name in MODEL_BASKET:
        summary_basket[model_name] = generate_summary(document, model_name)

    return summary_basket

def generate_summaries_custom(document):
    summary_basket = {}
    summary_basket[CUSTOM_MODEL_NAME] = generate_summary_custom_model(document, CUSTOM_MODEL_NAME)

    return summary_basket


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage", sys.argv[0], "<DocumentFile>")
        sys.exit(1)
    with open(sys.argv[1]) as fh:
        lines = fh.read()
        print(lines)
        print(generate_summaries(lines))
