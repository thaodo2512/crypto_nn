# Secrets Directory

This folder stores local-only secrets that should never be committed.

Steps
- Copy `.env.example` to `.env` and set your values:
  - `cp secrets/.env.example secrets/.env`
  - Edit `secrets/.env` and set `COINGLASS_API_KEY`

How to load
- macOS/Linux (bash/zsh):
  - `set -a; source secrets/.env; set +a`
  - Then run: `python ingest_cg.py ingest coinglass --conf conf/p1_inputs_cg.yaml`
- Windows PowerShell:
  - `Get-Content secrets/.env | ForEach-Object { if ($_ -match '^(.*?)=(.*)$') { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') } }`
  - Then run the CLI command in the same session.

Notes
- `.gitignore` excludes this directory except for `.env.example` and this README.
- The code reads `COINGLASS_API_KEY` from the environment and sends it as header `CG-API-KEY`.

