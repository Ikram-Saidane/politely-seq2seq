#!/bin/bash
# Train and evaluate every configuration, one after the other (CPU: about 3 hours in total).
export PYTHONPATH=src
python -m politely.evaluate --baselines
for c in A_lstm B_attention C_copy C_copy_nodrop D_gru_attention E_attention_h128 E_attention_h512 F_attention_tf0.5; do
  python -m politely.train --config configs/$c.json > experiments/$c.stdout 2>&1 || { echo "FAILED $c"; continue; }
  python -m politely.evaluate --models $c >> experiments/eval.log 2>&1
  echo "DONE $c"
done
for d in beam3 beam5 sample_t0.7 topk5; do python -m politely.evaluate --models C_copy C_copy_nodrop --decoding $d --splits valid; done
python -m politely.evaluate --models C_copy_nodrop --decoding beam5 --splits test test_unseen challenge
python -m politely.figures --best C_copy_nodrop
echo ALL_DONE
