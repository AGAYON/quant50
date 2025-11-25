# Quant50 🚀

Automated 50-asset portfolio management system using Alpaca API, LightGBM, and cvxpy.

## 🌟 Features
- **Automated Pipeline**: Ingestion → Feature Engineering → ML Prediction → Optimization → Execution → Reporting.
- **Machine Learning**: Weekly training using LightGBM with purged k-fold cross-validation.
- **Optimization**: Robust portfolio construction using Ledoit-Wolf shrinkage and cvxpy.
- **Reporting**: Daily PDF reports with PnL, drawdown, and turnover metrics.
- **API**: FastAPI endpoints for manual control and monitoring.

## 🛠️ Setup

### 1. Prerequisites
- Python 3.10+
- Alpaca Paper Trading Account (API Key & Secret)

### 2. Installation
```bash
# Clone repository
git clone <repo-url>
cd quant50

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```ini
ALPACA_API_KEY_ID=your_api_key
ALPACA_API_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets/v2
REPORTS_DIR=./data/reports
LOG_LEVEL=INFO
```

## 🚀 Usage

### Manual Execution (Local)
You can run the pipeline components manually:

```bash
# Run the full daily pipeline (Data -> Model -> Trade -> Report)
python jobs/daily_run.py

# Run weekly model training
python jobs/weekly_train.py
```

### API Server
Start the FastAPI server to trigger runs or get reports via HTTP:
```bash
uvicorn app.main:app --reload
```
- **POST /run**: Trigger daily pipeline.
- **POST /report**: Generate PDF report.
- **GET /orders**: View recent orders.

## 🤖 Automation (GitHub Actions)
The project is configured to run automatically on GitHub:

1. **Daily Run**: Executes Mon-Fri at 13:00 UTC (30 mins before market open).
2. **Weekly Train**: Retrains the model every Sunday at 00:00 UTC.

### Setting up Secrets
To enable automation, add these secrets in your GitHub Repo > Settings > Secrets > Actions:
- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`
- `ALPACA_BASE_URL` (optional, defaults to paper)

### 📧 Receiving Reports
Currently, the daily PDF report is uploaded as a **GitHub Artifact**.
1. Go to the **Actions** tab in your repo.
2. Click on the latest "Daily Run" workflow.
3. Scroll down to **Artifacts** and download `daily-report`.

*(Note: To receive reports via email, you would need to configure an SMTP step in `.github/workflows/daily_run.yml` or use a Discord webhook).*
