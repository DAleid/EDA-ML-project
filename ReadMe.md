# Stack Overflow 2025 AI Productivity Dashboard

A Streamlit dashboard for the Stack Overflow Developer Survey 2025. It includes interactive filters, EDA charts, relationships between AI adoption and perceived productivity, and a reproducible Logistic Regression benchmark.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 3000 --server.address 0.0.0.0
```

The app expects `survey_results_public.csv` next to `app.py`. The dashboard labels the outcome as perceived productivity because the survey is self-reported and does not contain measured work-output logs.
