# HospitaLens AI

**Predict · Detect · Simulate · Optimize**

HospitaLens AI is a hackathon prototype for demonstrating hospital operations planning with generated data. It forecasts admissions, converts demand into resource requirements, flags threshold breaches, and provides a deterministic Resource Decision Engine for what-if planning. It is not a clinical system and has not been validated for real operations.

## Problem and solution

Operations teams need to see how demand changes could affect beds, staff and diagnostic capacity. This prototype combines a Random Forest admissions forecast with explicit resource conversion assumptions, capacity calculations, bottleneck ranking, threshold alerts and explainable recommendations. The What-If Simulator sends inputs to FastAPI; the returned scenario response is kept temporarily in browser local storage and drives the existing Dashboard cards. It never writes scenario values to the hospital database. Return to Baseline removes that saved result.

## Architecture and technology

React and Vite render the UI. `frontend/src/services/api.js` is the single HTTP client. FastAPI owns the dataset, forecast, resource calculations, simulator, alerts and decisions. SQLite stores the operational history and configured capacity table. Scikit-learn trains one Random Forest per admission target.

```text
frontend/src/App.jsx ──> frontend/src/services/api.js ──HTTP──> backend/main.py
  Dashboard / Simulator                                    │
  temporary scenario state <── simulation JSON            ├─ forecast and evaluation
                                                           ├─ resource calculations
                                                           ├─ alerts and decision rules
                                                           └─ SQLite + synthetic CSV
```

## Data and limits

`backend/data/hospital_data.csv` contains 365 daily synthetic records, with dates checked for continuity and generated nonnegative values. If the CSV is absent, the backend generates it using NumPy seed `20261007`. The generator uses a rolling date window ending yesterday, so the seed fixes the random stream but the calendar dates (and weekend alignment) can change if regenerated later. The source includes admissions, discharges, occupancy, staffing availability and diagnostics; it is not patient or hospital data.

Resource capacities and conversion ratios are demonstration assumptions in `backend/main.py`. On the latest synthetic day, doctors and nurses are available-capacity limits; fractional calculated staff requirements are labeled FTE. General/ICU bed demand and diagnostic points can also be fractional model estimates. Returned capacity pressure is calculated from the returned one-decimal requirement and capacity; remaining capacity is capacity minus that same requirement. Pressure above 100% means projected requirement exceeds available capacity and is intentionally not clamped. The utilization-named API fields remain for contract compatibility. These ratios do not describe validated staffing standards.

## Forecast model and evaluation

The target variables are daily emergency, general and ICU admissions, modeled independently. Each Random Forest uses day of week, month, one-day lag, seven-day lag, and the prior seven-day mean. Lag and rolling features are shifted so a target day is not used to predict itself. Missing feature rows are dropped. The forecast endpoint recursively feeds predictions back as lags for the seven-day outlook.

The reported evaluation is a chronological 80/20 split of feature rows (last 72 days held out for the current 365-row dataset). It measures one-step predictions using observed prior lags; it is not the recursive seven-day forecast and does not validate real-world performance. Current holdout metrics, in admissions per day, are:

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| Emergency admissions | 3.114 | 3.879 | -0.272 |
| General admissions | 3.932 | 4.685 | 0.082 |
| ICU admissions | 0.890 | 1.068 | -0.103 |

The negative R² values mean the model performs worse than a holdout-mean predictor for those targets on this synthetic slice. Metrics are computed by the backend at runtime; no real-world accuracy claim is made.

## Resource Decision Engine

The page named Optimizer presents a deterministic rule engine, not a mathematical optimizer. It compares backend-projected requirements with configured capacity, ranks pressure, emits threshold alerts, and creates action prompts with the values and thresholds that triggered them. The Dashboard, Optimizer, Simulator and Resources use backend outputs. The Forecast page remains the baseline model forecast.

## What-If Simulator

`POST /api/simulation` validates bounded changes to emergency admissions, general admissions, nurse availability, doctor availability and ICU bed capacity. The same backend resource calculation then recalculates requirements, capacities, capacity pressure, remaining capacity, status, bottleneck, alerts and recommendations. The response contains both baseline and simulated resource arrays and a comparison against baseline forecast. `POST /api/emergency-surge` calls the same calculation with emergency admissions +30%.

The frontend stores the latest complete response under `hospitalens.activeScenario`. It survives navigation and refresh in that browser. Incomplete or outdated stored responses are discarded. Starting a new run clears the prior result while the request is pending; a failed request therefore cannot leave a stale scenario displayed. Return to Baseline clears the saved state. Scenario data is not persisted to SQLite.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` and `/api/health` | Service and synthetic-data health status |
| GET | `/api/dashboard` | Baseline resources, forecast, alerts, bottleneck and recommendations |
| GET | `/api/forecast` | Next-day and seven-day forecasts, plus holdout metrics |
| GET | `/api/resources` | Baseline resource values and bottleneck |
| GET | `/api/alerts` | Baseline threshold alerts |
| GET | `/api/recommendations` | Baseline rule-based recommendations |
| POST | `/api/simulation` | Validated what-if calculation and full response |
| POST | `/api/emergency-surge` | Emergency +30% preset through the same engine |

Interactive API documentation is available at `/docs` when the backend is running.

## Run locally

Requirements: Python 3.10+, Node.js 20.19+ (or 22.12+), and pnpm.

Backend, from the project root in PowerShell:

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PORT = "8000"
$env:CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
python run.py
```

In a second terminal:

```powershell
cd frontend
pnpm install
$env:VITE_API_BASE_URL = "http://localhost:8000"
pnpm run dev
```

Open `http://localhost:5173`. The first backend startup initializes the CSV (if missing) and SQLite tables. The database is local runtime data and is ignored by Git.

## Deployment

Deploy the FastAPI service from the `backend` directory with `python run.py`. It binds to `0.0.0.0` and uses the hosting platform's `PORT` value (default 8000). Set `CORS_ORIGINS` to the exact deployed frontend origin or comma-separated list of approved origins; CORS is not authentication. Credentials are disabled because this prototype has no authentication.

Build the static frontend from `frontend` and publish `frontend/dist`. Set `VITE_API_BASE_URL` to the deployed FastAPI base URL at build time. This `VITE_` value is public and must contain only the API origin, never a secret. A production build without this setting will show an API configuration error rather than silently call localhost. No hosting provider or deployed backend URL is configured in this source tree, so deployment still requires those platform settings.

For a production release, provide persistent storage for the SQLite database and CSV or mount a durable data volume. This demo has no authentication, authorization, audit trail, privacy controls or high-availability configuration.

## Source walkthrough

- `frontend/src/main.jsx`: React entrypoint and router mounting.
- `frontend/src/App.jsx`: navigation, page components, scenario local storage, simulator input/run flow and Dashboard rendering.
- `frontend/src/services/api.js`: base URL, HTTP error handling and endpoint mapping.
- `backend/main.py`: FastAPI routes, database initialization, synthetic data, feature engineering, Random Forest forecast/evaluation, resource math, simulation, alerts and decision rules.
- `backend/run.py`: deployment entrypoint that binds to `0.0.0.0:$PORT`.

## Safety and future work

This is a synthetic hackathon demonstration for hospital operations planning only. It must not be used for clinical diagnosis, care decisions or autonomous allocation. Real deployment would require governed real-world validation, privacy and security review, operational approval, authentication, auditability and locally verified capacity assumptions.
### Configured capacities and demonstration ratios

Default capacity configuration is 100 general beds, 20 ICU beds, 25 doctors, 80 nurses and 100 diagnostic points. The displayed doctor and nurse capacity is limited by the latest synthetic availability record. Current configured capacity rows are stored in SQLite and read by the backend.

| Conversion | Ratio |
|---|---:|
| General admission to general-bed demand | 0.55 beds |
| Emergency admission to general-bed demand | 0.18 beds |
| ICU admission to ICU-bed demand | 0.45 beds |
| Occupied bed to nursing requirement | 0.82 FTE |
| Daily admission to doctor requirement | 0.31 FTE |
| Incremental admission to diagnostic pressure | 1.45 points |

All ratios are prototype assumptions, not clinical or workforce planning standards.


## Build note

The current Vite production build passes. Vite reports one chunk-size warning: the main JavaScript bundle is about 653.8 kB minified (195.7 kB gzip), largely because Recharts and the UI are shipped together. This is understood; no new dependency was added and the warning does not block the build. A future code-splitting pass can reduce initial transfer size if the demo grows.
