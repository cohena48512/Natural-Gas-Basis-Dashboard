# Natural Gas Market Intelligence Dashboard

## Run locally
1. Install Python.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Upload an Excel (`.xlsx` / `.xls`) or CSV file in the sidebar.

## Required data columns
- Hub
- State
- Region
- Price After Basis
- Basis Differential

Optional:
- Date (or Month / Pricing Date / Effective Date)

## Notes
- The app accepts state names or two-letter state abbreviations.
- If State contains multiple states separated by `/`, it uses the first state, matching the original Colab workflow.
- Regional averages are calculated from the currently filtered rows.
- IQR outlier flags are screening indicators only, not proof of a market event.
- With one month of data, this version focuses on cross-sectional comparisons rather than time-series analysis.

## Sharing
For a quick handoff, send the folder or ZIP to someone who can run Python.
For a browser-based shareable link, deploy the app to Streamlit Community Cloud or an approved company hosting environment. Do not upload confidential client data to a public hosting service.
