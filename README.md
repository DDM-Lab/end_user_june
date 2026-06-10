# LLM End User

LLM-subject simulator for the DDMLab phishing-susceptibility experiment.
One run = one LLM "subject" iterates a fixed sequence of 56 emails in a
single chat session, picks an action per email, and self-reports the
substrings that drove its choice — the basis for the saliency vector used
downstream against human subjects.

See `docs/Experiment_Protocol.pdf` for the full protocol.

## Quickstart

```bash
pip install -e .
cp .env.example .env             # then fill in GEMINI_API_KEY / OPENAI_API_KEY
python scripts/build_sequences.py # generate data/sequences/sequence_{1..5}.txt
python run_experiment.py          # run sequence 1 on Gemini
```

The default `config.yaml` runs sequence 1 against `gemini-2.5-flash`, holds
one chat session across all 56 turns (so trial 56 sees trials 1..55 in
context), and appends rows to `data/dataset.csv`.

## Layout

```
config.yaml              central config (provider, sequence, temperature=0)
run_experiment.py        CLI entrypoint
scripts/
  build_sequences.py     build sequence_{1..5}.txt from Emails.csv + annotations
llm_end_user/
  experiment_manager.py  orchestrator: one chat session, loop 56 trials, write CSV
  providers/             ChatSession surfaces for gemini / openai / ollama
  saliency.py            self-reported saliency → per-token 0/1 vector
  prompts.py             system prompt + per-trial user message builder
  sequence_loader.py     parse data/sequences/sequence_N.txt (TSV)
  email_loader.py        load emails from data/Emails.csv (HTML → readable text)
  dataset_writer.py      append-only CSV sink
  types.py               shared payload schema
data/
  Emails.csv             200 emails (HTML body) — source of truth
  cognitive-phishing-annotations-latest.csv   per-email bias labels (human raters)
  sequences/             sequence_1.txt … sequence_5.txt (built from the two CSVs)
  dataset.csv            output (gitignored)
```

## Data

`data/Emails.csv` holds 200 emails (HTML bodies, sender, subject, type, author).
`data/cognitive-phishing-annotations-latest.csv` is the per-email annotation
file from human raters — each row is one user tagging one highlighted substring
with one bias. `scripts/build_sequences.py` joins these to pick 56 emails that
fill the 7 (bias) × 2 (author) × 2 (legitimacy) = 28 cells with 2 emails each,
then shuffles into 5 sequences.

## Conversation memory

The orchestrator opens one chat session per subject and pushes every email
through it as a user turn. The provider keeps the full message history alive,
so when the model answers email 56 it has seen emails 1..55 and its own
prior replies in context. This is by design — RQ3 in the protocol asks
whether subjects exhibit sequential effects.

## CLI

```bash
python run_experiment.py \
  --sequence 2 \
  --provider gemini \
  --model gemini-2.5-flash \
  --subject-id llm-run-042 \
  --dataset-path data/dataset_seq2.csv
```

CLI flags override `config.yaml`.

## Output

One CSV row per trial; columns: `subject_id, provider, model, sequence_id,
trial_order, email_id, bias, author, legitimacy, action, hit,
saliency_strings, saliency_vector, raw_response`. `saliency_strings` and
`saliency_vector` are JSON-encoded.
