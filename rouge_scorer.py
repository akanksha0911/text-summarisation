#!/usr/bin/env python3
from rouge_score import rouge_scorer


def score_summary(summary1, summary2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(summary1, summary2)

if __name__ == '__main__':
    print(score_summary("The night is black", "Moonless night"))