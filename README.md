# Football Similarity Explorer

An interactive football analytics project by **Vir Kapur**. It represents players as statistical vectors, finds similar profiles with cosine similarity, compares players with human-defined tactical archetypes, and visualizes the shared space with PCA.

![Python](https://img.shields.io/badge/Python-3.10%2B-263630)
![FastAPI](https://img.shields.io/badge/API-FastAPI-4f7168)
![Tests](https://img.shields.io/badge/tests-14%20passing-c86d4c)

## What it does

- Finds the players whose statistical profiles most closely resemble a selected footballer.
- Searches the dataset using tactical archetypes such as Poacher, False 9, Advanced Playmaker, and Ball-Playing Centre Back.
- Shows the underlying cosine value instead of presenting similarity as probability or player quality.
- Provides position-relative percentile profiles and side-by-side comparisons for up to three players.
- Projects the multidimensional player space into an interactive 2D or 3D PCA view.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/virkapur22/football-similarity-explorer.git
cd football-similarity-explorer

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Deploy to Vercel

Import this repository at [vercel.com/new](https://vercel.com/new) and keep these settings:

- Application Preset: **FastAPI**
- Root Directory: leave blank
- Build Command: leave blank
- Output Directory: leave blank

Vercel reads the FastAPI entrypoint from `pyproject.toml` and deploys the API and existing frontend as one Python Function. The project pins Python 3.12 through `.python-version`.

You can also deploy from the command line:

```bash
npm install --global vercel
vercel
vercel --prod
```

## How it works

### 1. Data preparation

The project uses the included 2024/25 FBref-derived export at:

```text
data/raw/fbref_kaggle/players_data_light-2024_2025.csv
```

Transfer duplicates are collapsed to each player's highest-minutes club row. Players below 900 minutes are excluded, and counting statistics are converted to per-90 rates.

### 2. Player vectors

Raw statistics are robust-scaled inside broad position groups (GK, DF, MF, and FW). Correlated statistics are then combined into interpretable football dimensions such as goal threat, creativity, progression, dribbling, defending, aerial ability, and goalkeeper distribution.

### 3. Player similarity

Player-to-player similarity is calculated with cosine similarity in the full position-aware dimension space:

```text
similarity = cosine(player_vector_a, player_vector_b)
display score = max(0, similarity) × 100
```

PCA coordinates are used only for visualization; they are not used to rank similar players.

### 4. Archetype similarity

Archetypes are transparent, human-defined statistical prototypes stored centrally in `src/model/archetypes.py`. Each target percentile is converted into the same normalized coordinate space as eligible player vectors before cosine similarity is calculated.

Archetypes are not machine-learned classes, probabilities, or player-quality ratings. A player's highest score is shown as the primary archetype, but all eligible matches remain available.

### 5. PCA visualization

PCA compresses the shared player-vector space into three display dimensions. The interface can show either the 3D projection or a simpler 2D view.

## Project structure

```text
frontend/                 Static application UI
src/api/                  FastAPI routes and static-file server
src/clean/                Minutes filtering and missing-value handling
src/explain/              Dimension-level match explanations
src/features/             Per-90, position, and football-dimension features
src/ingest/               FBref dataset loader
src/model/                Vectors, cosine similarity, PCA, and archetypes
scripts/                  Dataset inspection utility
tests/                    Model and API regression tests
config_fbref_kaggle.yaml  Dataset mapping and cleaning configuration
config_model.yaml         Scaling, dimension, PCA, and position-weight config
```

## API

FastAPI exposes interactive documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

| Route | Purpose |
| --- | --- |
| `GET /search?q=...` | Player-name search |
| `GET /player/{name}` | Full player profile |
| `GET /similar/{name}` | Player-to-player similarity |
| `GET /map/{name}` | PCA map and nearest-player connections |
| `GET /explain?player_a=...&player_b=...` | Dimension-level comparison |
| `GET /archetypes` | Archetype catalogue |
| `GET /archetypes/{id}` | One archetype definition |
| `GET /archetype-similarity/{name}` | All eligible archetype matches for a player |
| `GET /archetype-options` | Scout filter options |
| `GET /archetype-search` | Paginated archetype search |

## Tests

Install the development dependencies and run the suite:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

The tests cover vector construction and ordering, normalization parity, direct cosine calculations, position eligibility, archetype ranking, pagination, profile-specific metrics, legacy-route removal, and the approved Scout UI structure.

## Model limitations

- Detailed position families are inferred because the source provides broad position labels.
- Per-90 statistics do not describe tactics, opponent strength, tracking, event locations, contracts, or market value.
- Archetype targets encode documented football judgment and should be reviewed when the dataset changes.
- The 900-minute threshold improves stability but excludes small-sample players.
- Cosine similarity describes profile direction more than absolute output and should not be treated as player quality.
- A single-season dataset can reflect temporary form or role changes.

## Data note

The repository includes the dataset required to run the project locally. Before redistributing it publicly, confirm that the source dataset's license permits inclusion in your GitHub repository.
