#!/usr/bin/env python3

import sys
import argparse
import extractive_summary as es
import abstractive_summaries as ase
from rouge_score import rouge_scorer


def score_summary(summary1, summary2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return scorer.score(summary1, summary2)["rougeL"].fmeasure


def best_summary(document):
    extractive = es.generate_summary(document, 2)
    abstractives = ase.generate_summaries(document)

    score_board = {}
    best_score = {"score" : 0, "summary" : "", "model_name": ""}
    for model_name in abstractives:
        score_board[model_name] = score_summary(extractive, abstractives[model_name])
        if score_board[model_name] > best_score["score"]:
            best_score["score"] = score_board[model_name]
            best_score["summary"] = abstractives[model_name]
            best_score["model_name"] = model_name
    return best_score["summary"], best_score["model_name"], score_board, extractive, abstractives

def parse_input():
    DESC = """
    Usage: %(prog)s [options] Document
    """
    EXAMPLES = """
    eg: python3 %(prog)s articles/msft.txt
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESC,
                                     epilog=EXAMPLES)
    parser.add_argument('-v', '--verbose', help='Detailed Report', default=False, action="store_true")
    parser.add_argument('document', help='Article to summarize')
    if len(sys.argv) < 2:
        parser.parse_args(['-h'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_input()
    with open(args.document) as fh:
        lines = fh.read()
        summary, model_name, sb, ex, ab = best_summary(lines)
        if args.verbose:
            print(lines)
            print("\nExtractive Summary\n", ex)
            print("\nAbstractive Summaries")
            for mn in ab:
                print(mn)
                print(ab[mn])
            print("\nScore Board")
            for mn in sb:
                print(mn, "=", sb[mn])
            print("=" * 120)
            print("Best Summary from", model_name)
            print("=" * 120)
        print(summary)
