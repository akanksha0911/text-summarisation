#!/usr/bin/env bash

if [ $# -lt 1 ]
then
  echo "Usage $0 <document>"
  exit
fi
filename=$(basename $1)
python3 best_summary.py -v $1 > reports/$filename
tail -1 reports/$filename