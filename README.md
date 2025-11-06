
# Time Series Anomaly Detection for IoT Sensors

## Structure
- `data/` - place dataset CSV(s) here
- `models/` - saved scaler and trained models
- `results/` - saved plots and CSV outputs
- `src/` - source modules
- `run_pipeline.py` - main script to run full pipeline
- `requirements.txt` - python dependencies

## How to run
1. Activate venv: `.venv\Scripts\Activate.ps1`
2. Install (if not installed): `pip install -r requirements.txt`
3. Place dataset CSV(s) in `data/`
4. Run the pipeline: `python run_pipeline.py`
5. Results saved to `results/` and models to `models/`

## Notes
- LSTM autoencoder is trained on the first 50% of the dataset (assumed normal) — document this assumption in your summary.
