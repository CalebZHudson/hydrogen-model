# Subsidising Green Hydrogen: domestic hydrogen market model

Interactive version of the domestic-market section of the model in *Subsidising Green Hydrogen: A Multi-Market
Equilibrium Model for Australia* (Caleb Hudson, Qazi Haque and Firmin Doko Tchatoka, University of Adelaide).

SMR and grid-connected electrolysers compete to meet domestic hydrogen demand. Choose a support instrument, change any
input and press **Solve**. Each instrument has the same A$6.7 bn envelope over FY2027-28 to FY2039-40.

## Files

| File | What it is |
|---|---|
| `index.html` | Loads the app in the visitor's browser (used by GitHub Pages) |
| `app.py` | The app's page: inputs, charts and tables |
| `h2_model.py` | The model: linear programmes solved with HiGHS (SciPy) |
| `requirements.txt` | Python packages (used by Streamlit Cloud or your own computer) |

## Run on your own computer

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
