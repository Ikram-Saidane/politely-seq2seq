# Politely — rewriting impolite sentences with an encoder–decoder built from scratch

NLP course project · *Text generation with encoder–decoder architectures* · Ikram Saidane

**Deliverables:** the notebook [`NLP_Project.ipynb`](NLP_Project.ipynb), which goes through the whole
project step by step, and the 2-page report [`NLP_Project_Report.docx`](NLP_Project_Report.docx) (Word; PDF version: [`NLP_Project_Report.pdf`](NLP_Project_Report.pdf)).

## 1. Presentation and objective

Politely takes a blunt, impatient or dismissive English sentence and generates a polite sentence
that says **the same thing**. The objective of the project is to build the complete neural
text-generation pipeline ourselves: preprocessing → vocabulary → tokenisation → representation →
encoder → decoder → training → generation → evaluation. No pretrained model (GPT, T5, BART…) and no
API is used: the encoder–decoder is implemented with PyTorch and trained from scratch.

| Impolite input | Polite output (final model) |
|---|---|
| Close the window now. | Would you mind closing the window as soon as possible? |
| You're wrong, the meeting is on Tuesday. | I might be mistaken, but I believe the meeting is on Tuesday. |
| Turn the music down. | Could I ask you to turn the music down? Thank you! |

## 2. The NLP problem

It is a **paraphrase generation / style transfer** task, formulated as sequence-to-sequence
generation: the model learns P(y | x) = Π P(y_t | y_<t, x), where x is the impolite sentence and y
the polite one. A good output must be:

- **polite** (tone),
- **faithful**: the action, object, person, time and numbers of the input are kept,
- **fluent**.

## 3. The dataset

**Source.** Public parallel corpora close to this task (e.g. ParaDetox) are full of insults, so the
dataset was built for the project. The program `data/generate_dataset.py` combines building
blocks written by hand in `data/resources.py`: situations, objects, people, times, and 108 impolite
and 54 polite phrasings over 9 categories. It is reproducible: seed 42 always rebuilds the same
dataset. The impolite side is rude only by its tone. There is no insult or swear word, and every
sentence is checked against a list of banned words.

**Fields.** `impolite` (input) → `polite` (target). Each row also has `id`, `split`, `category`,
`template_id` and `ref_index`. Each impolite input has 2 polite references in train and 3 in the
evaluation sets.

**Categories (balanced).** request, reminder, refusal, criticism, complaint, disagreement, question,
stop, announcement.

| Split | Impolite inputs | Pairs | Used for |
|---|---|---|---|
| train | 14,400 | 28,800 | training |
| valid | 1,350 | 4,050 | choosing the model and the decoding strategy |
| test | 1,350 | 4,050 | test, same phrasings as training |
| test_unseen | 900 | 2,700 | test, impolite phrasings **never seen** in training |
| challenge | 72 | 72 | test, sentences **written by hand** |

The splits are made **by content**, so no sentence appears in two splits.

**Improved once.** The first version (`data_v1/`) covered only office life (1,371-word vocabulary).
It contained incoherent pairs such as *"Move your car out of the library"*, and its models reached
0–1 % success on hand-written sentences. In version 2 (`data/`), I removed those pairs and widened
the vocabulary to everyday life (1,805 words). The comparison is in `reports/dataset_v1_vs_v2.json`.

## 4. Steps of the project

1. Build the dataset and check it (`data/generate_dataset.py`, `data/analysis.py`).
2. Preprocessing and vocabulary (`preprocessing.py`).
3. Encoder–decoder models A, B and C plus variants (`model.py`).
4. Sanity check: each model must memorise 40 pairs. Then full training (`train.py`).
5. Generation with greedy, beam search and sampling (`inference.py`).
6. Evaluation against two baselines on three test sets, plus a human evaluation (`evaluate.py`).
7. Figures, report and interface (`figures.py`, `reports/`, `app.py`).

## 5. Preprocessing and tokenisation (`preprocessing.py`)

| Step | Why |
|---|---|
| Unicode normalisation (NFKC, simple quotes) | the same character always gives the same token |
| Lowercasing | "Could" and "could" become one word, so there are fewer words to learn. Case is restored for display only |
| Punctuation as separate tokens | "?" matters: polite requests are usually questions |
| Word tokenisation, contractions kept (*don't*) | simple and enough for short English sentences |
| Vocabulary on **train only**, words seen ≥ 2 times | no information from the test sets leaks into training |
| Special tokens `<PAD>`=0 `<SOS>`=1 `<EOS>`=2 `<UNK>`=3 | padding, start and end of the output, unknown words |
| Truncation at 40 tokens, padding per batch | tensors of the same length. `<PAD>` is masked in the encoder, in the attention and in the loss |

Example: `Close the window now.` → `close · the · window · now · .` → `390 4 573 58 5 2`.
Each id then becomes a 128-dimensional embedding vector.

## 6. The encoder–decoder architecture (`model.py`)

```
input ids → Embedding (128) → LSTM encoder (1 layer, unidirectional, hidden 256)
                                   │ one state per input word + final state
                                   ▼
<SOS> → Embedding → [ Bahdanau attention → context vector ] → LSTM decoder (hidden 256)
                                   → Linear → softmax over 1,805 words (+ copy) → next word … until <EOS>
```

- **Encoder.** It reads the sentence and returns one hidden state per word. Its final state is the
  initial state of the decoder.
- **Decoder.** At each step it receives the previous word (and the context vector) and predicts the
  next word. It starts with `<SOS>` and stops at `<EOS>` or after 40 words.
- **Attention (model B).** A single vector forgets parts of long inputs. With attention, at each step
  the decoder scores every input state: score_j = vᵀ tanh(W_enc h_j + W_dec s). The softmax turns
  the scores into weights, and the context vector c = Σ α_j h_j is given to the decoder.
- **Copy mechanism (model C).** p_gen = σ(w·[c; s; e]) mixes "generate a vocabulary word" with
  "copy an input word", so names, numbers and unknown words are kept.

| Model | Description | Parameters |
|---|---|---|
| A | LSTM encoder–decoder, no attention (baseline) | 1.30 M |
| B | A + attention | 1.66 M |
| **C** | **B + copy mechanism (final)** | **1.67 M** |
| C' | C + word dropout | 1.67 M |
| D | B with GRU cells | 1.40 M |
| E | B with hidden size 128 / 512 | 0.88 / 4.42 M |
| F | B with teacher forcing 0.5 | 1.66 M |

## 7. Training (`train.py`)

| | |
|---|---|
| Loss | token-level cross-entropy, `<PAD>` ignored |
| Optimiser | Adam, learning rate 0.001, halved when the validation loss stops improving |
| Batch size / dropout / clipping | 64 / 0.3 / 1.0 |
| Embedding / hidden dimension / layers | 128 / 256 / 1 |
| Teacher forcing | 1.0 (0.5 for model F) |
| Epochs | at most 30, early stopping after 3 epochs without improvement on validation |
| Hardware | CPU, about 1 to 1.5 minutes per epoch |

The training and validation losses are saved for each epoch (`experiments/<model>/history.json`
and `curves.png`). The best model on validation is saved in `checkpoints/<model>.pt`.

## 8. Evaluation metrics (`evaluate.py`)

Several metrics are needed, because one metric alone does not say if a rewrite is good:

- **BLEU** (multi-reference, written by hand and checked against sacrebleu) and **chrF**: closeness
  to the references.
- **Exact match**: the output is exactly one of the references.
- **Content kept**: % of the content words of the input still present in the output.
- **Politeness**: probability given by a small politeness classifier (`politeness.py`, logistic
  regression trained on train only).
- **Success**: polite **and** all content words kept **and** no repeated word. This is the main score.
- **Token accuracy** and **perplexity** (with teacher forcing) as diagnostics.
- **Human evaluation** of the final system: acceptable / partial / failed. There was one annotator
  only, and the labels were produced with the help of an AI assistant.

The two baselines are **copy** (output = input) and **rules** (remove rude words, add "please").

**Results** (success rate, all metrics in `reports/results.md`):

| System | test | test_unseen | challenge |
|---|---|---|---|
| Copy baseline | 0.000 | 0.034 | 0.000 |
| Rule-based baseline | 0.125 | 0.253 | 0.097 |
| A · LSTM, no attention | 0.845 | 0.350 | 0.111 |
| B · + attention | 0.912 | 0.582 | 0.222 |
| **C · + copy, beam 5 (final)** | **0.999** | **0.787** | **0.764** |
| C, judged acceptable by hand | 0.975 (40) | 0.375 (40) | 0.528 (72) |

Attention helps, but the biggest gain comes from the copy mechanism. A bigger model, GRU cells,
teacher forcing 0.5 or word dropout did not help. The model is excellent on sentences like its
training data, and about half of its rewrites of hand-written sentences are fully acceptable.

## 9. Project structure

```
politely-seq2seq/
├── NLP_Project.ipynb     the whole project step by step (explanations, code, results)
├── NLP_Project_Report.docx  the 2-page report (Word) + NLP_Project_Report.pdf
├── NLP_Project_Dataset.csv  the whole dataset in one file (all splits, with the "split" column)
├── preprocessing.py     normalisation, tokenisation, vocabulary, ids, padding, batches
├── model.py             hyper-parameters (Config), encoder, attention, decoder, copy mechanism
├── train.py             training loop, teacher forcing, early stopping, sanity check
├── inference.py         loading a saved model, greedy / beam search / sampling, command line
├── evaluate.py          metrics, baselines, evaluation of every system, results table
├── politeness.py        politeness classifier used as a metric
├── figures.py           figures of the report
├── app.py               Streamlit interface
├── data/
│   ├── generate_dataset.py   builds the dataset (versions 1 and 2)
│   ├── resources.py          hand-written building blocks of the dataset
│   ├── dataset_v1.py         rebuilds the first version and its preprocessing in data_v1/
│   ├── analysis.py           dataset statistics and figures
│   ├── train.csv valid.csv test.csv test_unseen.csv challenge.csv
│   ├── politeness_dataset.csv   all the pairs in one file
│   ├── dataset_card.json, vocab.json
├── data_v1/             first version of the dataset + its vocabulary and preprocessed file
├── configs/             one JSON file per model (A … F)
├── checkpoints/         trained models (*.pt) and the politeness classifier
├── experiments/         per model: training log, learning curves, metrics, predictions
├── archive_v1/          results of the first models trained on data_v1
├── reports/             report_detailed.pdf (17 pages), results, figures/
├── tests/               pytest tests (pipeline and interface)
├── run_app.bat          Windows: install and launch the interface
├── train_all.bat        Windows: rebuild the data, train and evaluate everything
└── requirements.txt
```

## 10. Technologies

Python 3.10+, **PyTorch** (model and training), NumPy and pandas (data), matplotlib (figures),
sacrebleu (chrF and BLEU check), **Streamlit** and Altair (interface), pytest (tests).

## 11. Installation

A GPU is not needed. Every model was trained on a CPU.

**Windows, one click:** double-click `run_app.bat`. The first time, it creates the virtual
environment `.venv`, installs the dependencies (a few minutes), then opens the interface.

**Manually (Windows PowerShell):**

```powershell
cd politely-seq2seq
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On Linux / macOS, activate with `source .venv/bin/activate` instead.

## 12. Running the project

All commands are run from the project folder, with the virtual environment activated.
The trained models are already in `checkpoints/`, so steps 3 and 4 work immediately.

```bash
# 1. data (optional: the dataset and the vocabulary are already in data/)
python data/generate_dataset.py      # dataset              -> data/*.csv
python data/dataset_v1.py            # first version         -> data_v1/
python data/analysis.py              # statistics + figures  -> reports/
python preprocessing.py              # vocabulary (train only) -> data/vocab.json
python politeness.py                 # politeness classifier -> checkpoints/politeness_clf.json

# 2. generate a polite sentence with the trained model
python inference.py "Close the window now."
python inference.py "Close the window now." --model B_attention --decoding greedy

# 3. evaluate
python evaluate.py --baselines
python evaluate.py --models C_copy_nodrop                       # greedy decoding, all splits
python evaluate.py --models C_copy_nodrop --decoding beam5      # beam search
python figures.py --best C_copy_nodrop

# 4. tests
python -m pytest

# 5. the notebook (about 10 minutes, run all cells from top to bottom)
jupyter notebook NLP_Project.ipynb
```

## 13. Training

```bash
python train.py --config configs/C_copy_nodrop.json --overfit 40   # sanity check: memorise 40 pairs
python train.py --config configs/C_copy_nodrop.json                # full training (about 30 minutes on CPU)
```

Replace the config to train another model (`A_lstm`, `B_attention`, `C_copy`, `D_gru_attention`,
`E_attention_h128`, `E_attention_h512`, `F_attention_tf0.5`). `train_all.bat` rebuilds the data,
then trains and evaluates the 8 models (about 3 hours on CPU). Training again overwrites the
checkpoint of that model.

## 14. User interface

```bash
streamlit run app.py
```

The interface opens at http://localhost:8501. Its tabs:

- **Rewrite**: type a sentence or choose an example. The page shows the polite version, the
  politeness before and after, the content words kept or lost, the attention map and the copy
  switch. You can choose the model and the decoding in the sidebar.
- **Compare models**: the same sentence rewritten by the two baselines and every model.
- **Training & results**: success rates, human evaluation, learning curves, decoding strategies
  and the figures of the report.
- **Dataset**: splits, balance, examples, vocabulary, version 1 vs version 2.
- **How it works**: the pipeline, a worked example with the real ids, and the models.

## 15. Examples of use

Real outputs of the final model (`python inference.py …`, beam search):

```
Close the window now.
  -> Would you mind closing the window as soon as possible?
You're wrong, the meeting is on Tuesday.
  -> I might be mistaken, but I believe the meeting is on Tuesday.
Give me my keys back, I need them right now.
  -> Could you please give me my keys back?
```

The last example shows a typical limit: the output is polite and keeps the request, but the
explanation "I need them" is lost. Successes and failures are analysed in the notebook and in `NLP_Project_Report.pdf`.
