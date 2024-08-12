"""Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')
  parser.add_argument('--data_file', metavar='data.json', help='Input data JSON file.')
  parser.add_argument('--pred_file', metavar='pred.json', help='Model predictions.')
  #parser.add_argument('--out-file', '-o', metavar='eval.json',
  #                    help='Write accuracy metrics to file (default is stdout).')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(lower(s)))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}
  for item in dataset:
    qid = item['question_id']
    gold_answer = item["answer"]
    if str(qid) not in preds:
        print('Missing prediction for %s' % qid)
        continue
    a_pred = preds[str(qid)]
    # Take max over all gold answers
    exact_scores[qid] = compute_exact(gold_answer, a_pred)
    f1_scores[qid] = compute_f1(gold_answer, a_pred)
  return exact_scores, f1_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
  total = len(exact_scores)
  out_eval = collections.OrderedDict([
      ('exact', 100.0 * sum(exact_scores.values()) / total),
      ('f1', 100.0 * sum(f1_scores.values()) / total),
      ('total', total),
  ])
  out_eval['overall'] = 0.5 * out_eval['exact'] + 0.5 * out_eval['f1']
  return out_eval


def main():
  with open(OPTS.data_file, encoding="utf-8") as f:
    dataset = json.load(f)
  with open(OPTS.pred_file, encoding="utf-8") as f:
    preds = json.load(f)
  exact_raw, f1_raw = get_raw_scores(dataset, preds)
  print ("EM:{}, F1:{}".format(exact_raw, f1_raw))
  out_eval = make_eval_dict(exact_raw, f1_raw)
  print (out_eval)


if __name__ == '__main__':
  OPTS = parse_args()
  main()
