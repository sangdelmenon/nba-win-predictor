# NBA Win Predictor

A machine learning application that predicts NBA game outcomes using team statistics, player impact scores, injury reports, and roster data. Features a live data pipeline and an interactive web interface.

## Features

- **Live Data Pipeline** — fetches current team stats, rosters, and injury reports
- **Feature Engineering** — player impact scores, rest days, home/away splits, recent form
- **ML Model** — trained classifier for win probability prediction
- **Interactive App** — web UI to select matchups and view win probabilities
- **Auto-updating** — roster and injury data refreshed automatically

## Project Structure

```
src/
├── agent.py            # Orchestrates prediction pipeline
├── data_collector.py   # Fetch team and game statistics
├── feature_engineering.py  # Build features for model input
├── model.py            # ML model definition and inference
├── player_impact.py    # Calculate player impact metrics
├── injury_fetcher.py   # Fetch and process injury reports
├── roster_updater.py   # Keep roster data current
data/                   # Cached datasets
models/                 # Trained model checkpoints
app.py                  # Web application entry point
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py
```

Then open `http://localhost:8501` (Streamlit) or `http://localhost:5000` (Flask) in your browser.

## Tech Stack

- **Language**: Python 3
- **ML**: scikit-learn / PyTorch
- **App**: Streamlit or Flask
- **Data**: NBA stats APIs
- **Libraries**: pandas, numpy, requests
