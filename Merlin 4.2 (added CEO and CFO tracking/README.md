# Merlin 4.0

Streamlit app for multi-asset ML with self-improvement, forum-buzz/volume radar, news features, and weekly trade plans.

## Quick start (Windows)

1) Install Python 3.10+.
2) `pip install -r requirements.txt`
3) (Optional) Set API keys in a `.env` file or env:
   - `ALPHA_VANTAGE_API_KEY`
   - `SERPAPI_API_KEY` (optional, for forum buzz)
4) Double-click `run_merlin_ui.bat` or run `python run.py`.

## Notes
- If you don't have a SerpAPI key, forum buzz still works but returns 0 counts until you provide a key.
- First run creates all tables in `stockmarket.db` automatically.
