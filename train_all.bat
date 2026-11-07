@echo off
REM Politely - rebuilds the data, trains and evaluates every model (CPU: about 3 hours).
cd /d "%~dp0"
set PY=.venv\Scripts\python -X utf8
if not exist .venv\Scripts\python.exe set PY=python -X utf8

REM 1. dataset, statistics, vocabulary, politeness classifier
%PY% data\generate_dataset.py
%PY% data\dataset_v1.py
%PY% data\analysis.py
%PY% preprocessing.py
%PY% politeness.py
%PY% evaluate.py --baselines

REM 2. sanity check + training + evaluation of each model
for %%c in (A_lstm B_attention C_copy C_copy_nodrop D_gru_attention E_attention_h128 E_attention_h512 F_attention_tf0.5) do (
  %PY% train.py --config configs/%%c.json --overfit 40
  %PY% train.py --config configs/%%c.json
  %PY% evaluate.py --models %%c
)

REM 3. decoding strategies (chosen on validation), final system on the test sets, figures
for %%d in (beam3 beam5 sample_t0.7 topk5) do %PY% evaluate.py --models C_copy C_copy_nodrop --decoding %%d --splits valid
%PY% evaluate.py --models C_copy_nodrop --decoding beam5 --splits test test_unseen challenge
%PY% figures.py --best C_copy_nodrop
pause
