# Impact of AI on Students: EDA and Streamlit Dashboard

Final Project, Unit 3: Individual Data Exploration & Dashboard Development.

**Dataset:** [Impact of AI on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students) (Kaggle, 50,000 rows x 16 columns).
The dataset page does not document how the data was collected; it looks very clean and balanced, so it may be simulated.
The report discusses this in Section 4.5.

## Project structure
```
.
├── EDA_AI_Impact_on_Students.ipynb   # full EDA + cleaning (creates clean_data.csv and figures/)
├── app.py                            # Streamlit dashboard (reads clean_data.csv)
├── clean_data.csv                    # cleaned dataset (created by the notebook)
├── clean_data_full.csv               # cleaned data + helper columns (used by the report builder)
├── build_report.py                   # builds the PDF report from the data and figures
├── figures/                          # plots saved by the notebook
├── screenshots/                      # your dashboard screenshots (inserted into the PDF)
├── AI_Impact_on_Students_Report.pdf  # final report
├── presentation.pptx                 # 5-7 minute presentation
└── requirements.txt
```

## How to run
1. **Install**
   ```bash
   pip install -r requirements.txt
   ```
2. **Get the data:** download the CSV from Kaggle and place it in this folder (or let the notebook download it with `kagglehub`).
3. **Run the notebook** `EDA_AI_Impact_on_Students.ipynb` (Run All). It creates `clean_data.csv`, `clean_data_full.csv` and `figures/`.
4. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```
   Take 3-4 screenshots (sidebar filters + a few tabs) and save them in `screenshots/`.
5. **Build the report**
   ```bash
   python build_report.py --name "Your Name" --raw path/to/original.csv --github https://github.com/you/repo
   ```
   Output: `AI_Impact_on_Students_Report.pdf`.

## Dashboard features
- Sidebar: dataset description, dropdown / multiselect / slider filters.
- Main page: KPI cards, data preview, summary statistics, interactive Plotly charts, insight text computed from the filtered data.
- Tabs: data and statistics, AI usage vs outcomes, burnout and well-being, groups and policy, correlations, chart explorer.

## Notes
All findings are associations in observational (possibly simulated) data; they do not show that AI causes the outcomes.
