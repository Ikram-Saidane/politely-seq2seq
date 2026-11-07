@echo off
REM Politely - regenerate the data, train and evaluate every model (CPU: about 3 hours).
cd /d "%~dp0"
set PY=.venv\Scripts\python
if not exist %PY% set PY=python
%PY% -m politely.generate_dataset
%PY% -m politely.analysis
%PY% -m politely.build_vocab
%PY% -m politely.politeness
%PY% -m politely.evaluate --baselines
for %%c in (A_lstm B_attention C_copy C_copy_nodrop D_gru_attention E_attention_h128 E_attention_h512 F_attention_tf0.5) do (
  %PY% -m politely.train --config configs/%%c.json
  %PY% -m politely.evaluate --models %%c
)
for %%d in (beam3 beam5 sample_t0.7 topk5) do %PY% -m politely.evaluate --models C_copy C_copy_nodrop --decoding %%d --splits valid
%PY% -m politely.evaluate --models C_copy_nodrop --decoding beam5 --splits test test_unseen challenge
%PY% -m politely.figures --best C_copy_nodrop
pause
