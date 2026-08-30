# Running the extraction as a Kaggle batch kernel

`scripts/kaggle_run.py` pushes `notebooks/run_on_kaggle.ipynb` to Kaggle, polls
until it finishes, and downloads the artifacts.

## What this is not

**Batch kernels are a different execution mode from the interactive notebook.**
There is no attach, no cell-by-cell, and no interrupt. A push starts a fresh
session that runs top to bottom and saves whatever is in `/kaggle/working`.

In particular the API cannot see, interrupt, or read variables out of a session
you have open in a browser. If a result exists only in an interactive kernel's
memory, this cannot rescue it.

There is also **no cancel command** in the CLI (`list files get init push pull
output status logs update delete topics` — no `cancel`). A run started here is
stopped from the web UI, not from the terminal.

## Cost

A run consumes GPU quota — 30 h/week on the free tier, and the measured
extraction is ~3.2 h. Pushing is therefore not free, and `kaggle_run.py push`
requires `--yes` so it cannot happen as a side effect of something else.

## Credentials

CLI 2.2.4 uses an API token, not the older `kaggle.json`:

1. https://www.kaggle.com/settings/api -> "Generate New Token"
2. Save it to `~/.kaggle/access_token`, or export `KAGGLE_API_TOKEN`

Then set your username in `kaggle/kernel-metadata.json` (`id` field) or export
`KAGGLE_USERNAME` and let the script substitute it.

The token is account credentials. Keep it out of the repo and out of chat;
`.gitignore` covers `kaggle/access_token` and `kaggle/kaggle.json`.
