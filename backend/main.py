from __future__ import annotations

import sqlite3
import os
from functools import lru_cache
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "hospital.db"
CSV_PATH = ROOT / "data" / "hospital_data.csv"
CAPACITIES = {"general_beds": 100, "icu": 20, "doctors": 25, "nurses": 80, "diagnostics": 100}
THRESHOLDS = ((95, "Critical"), (85, "High"), (70, "Moderate"), (0, "Normal"))
PLANNING_UTILIZATION_THRESHOLD = 85
RECOMMENDATION_UTILIZATION_THRESHOLD = PLANNING_UTILIZATION_THRESHOLD
ASSUMPTIONS = {"beds_per_general_admission": 0.55, "beds_per_emergency_admission": 0.18, "icu_beds_per_icu_admission": 0.45, "nurses_per_occupied_bed": 0.82, "doctors_per_admission": 0.31, "diagnostics_points_per_admission": 1.45}
DEMAND_COLS = ["emergency_admissions", "general_admissions", "icu_admissions"]
FORECAST_FEATURES = ("dow", "month", "lag1", "lag7", "mean7")
MODEL_PARAMS = {"n_estimators": 80, "max_depth": 6, "min_samples_leaf": 3, "random_state": 42, "n_jobs": 1}


def make_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Build one-step-ahead features using only calendar fields and prior observations."""
    work = df[["date", target]].copy()
    work["dow"] = work.date.dt.dayofweek
    work["month"] = work.date.dt.month
    work["lag1"] = work[target].shift(1)
    work["lag7"] = work[target].shift(7)
    work["mean7"] = work[target].shift(1).rolling(7).mean()
    return work.dropna()


def evaluate_forecast_models(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate one-step predictions on the final 20% chronological holdout."""
    metrics: dict[str, dict[str, float]] = {}
    holdout_days = 0
    for target in DEMAND_COLS:
        work = make_features(df, target)
        split = int(len(work) * 0.8)
        train, test = work.iloc[:split], work.iloc[split:]
        if train.empty or test.empty:
            raise ValueError(f"Not enough rows to evaluate forecast target: {target}")
        model = RandomForestRegressor(**MODEL_PARAMS).fit(train[list(FORECAST_FEATURES)], train[target])
        prediction = model.predict(test[list(FORECAST_FEATURES)])
        error = prediction - test[target].to_numpy(dtype=float)
        actual = test[target].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error ** 2)))
        denominator = float(np.sum((actual - actual.mean()) ** 2))
        r2 = float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else 0.0
        metrics[target] = {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 3)}
        holdout_days = max(holdout_days, len(test))
    return {"method": "Chronological 80/20 holdout; one-step rolling predictions use observed prior lags", "holdout_days": holdout_days, "metrics": metrics}


def make_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(20261007)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=365)
    rows, general_occ, icu_occ = [], 66.0, 13.0
    surge_days = {48, 49, 175, 176, 291, 292}
    for i, dt in enumerate(dates):
        weekend = dt.dayofweek >= 5
        season = 1 + .12 * np.sin(2 * np.pi * (dt.dayofyear - 25) / 365)
        surge = 1.55 if i in surge_days else 1.0
        emergency = max(1, int(rng.poisson((10.5 if weekend else 13.2) * season * surge)))
        general = max(1, int(rng.poisson((18 if weekend else 22) * season)))
        icu = max(1, int(round(emergency * .17 + rng.normal(1.8, .8))))
        discharges = max(1, int(round((emergency + general) * rng.uniform(.82, .99))))
        general_occ = float(np.clip(general_occ + (general - 21) * .30 + (emergency - 12) * .15 + (76 - general_occ) * .06 + rng.normal(0, 2), 48, 96))
        icu_occ = float(np.clip(icu_occ + (icu - 2.5) * .35 + (15 - icu_occ) * .08 + rng.normal(0, .7), 7, 20))
        staff_factor = rng.choice([.86, .92, .96, 1.0], p=[.06, .18, .35, .41])
        doctors = max(10, int(round(25 * staff_factor + rng.normal(0, .6))))
        nurses = max(40, int(round(80 * staff_factor + rng.normal(0, 1.5))))
        diagnostic = float(np.clip(38 + emergency * 1.0 + general * .9 + rng.normal(0, 5), 35, 94))
        rows.append({"date": dt.date().isoformat(), "day_of_week": dt.dayofweek, "month": dt.month,
                     "emergency_admissions": emergency, "general_admissions": general, "icu_admissions": icu,
                     "discharges": discharges, "general_bed_occupancy": round(general_occ), "icu_occupancy": round(icu_occ),
                     "doctors_available": doctors, "nurses_available": nurses, "diagnostic_utilization": round(diagnostic)})
    return pd.DataFrame(rows)


def initialize_db() -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        make_dataset().to_csv(CSV_PATH, index=False)
    df = pd.read_csv(CSV_PATH)
    required = {"date", "day_of_week", "month", *DEMAND_COLS, "discharges", "general_bed_occupancy", "icu_occupancy", "doctors_available", "nurses_available", "diagnostic_utilization"}
    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
    if (not required.issubset(df.columns) or len(df) < 300 or df[list(required - {"date"})].isna().any().any()
            or dates.isna().any() or dates.duplicated().any() or not dates.sort_values().reset_index(drop=True).equals(dates.reset_index(drop=True))
            or (df[list(required - {"date"})] < 0).any().any()
            or (dates >= pd.Timestamp.today().normalize()).any()
            or (df.day_of_week != dates.dt.dayofweek).any() or (df.month != dates.dt.month).any()):
        raise ValueError("Synthetic hospital dataset failed schema or completeness validation")
    if not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Synthetic hospital dataset dates must be continuous daily records")
    bounded = ((df.day_of_week > 6) | (df.month < 1) | (df.month > 12) | (df.emergency_admissions < 1) | (df.general_admissions < 1)
               | (df.icu_admissions < 1) | (df.discharges > df.emergency_admissions + df.general_admissions)
               | (df.general_bed_occupancy > CAPACITIES["general_beds"]) | (df.icu_occupancy > CAPACITIES["icu"])
               | (df.diagnostic_utilization > CAPACITIES["diagnostics"]))
    if bounded.any():
        raise ValueError("Synthetic hospital dataset contains out-of-range operational values")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS operational_records (date TEXT PRIMARY KEY, day_of_week INTEGER, month INTEGER, emergency_admissions INTEGER, general_admissions INTEGER, icu_admissions INTEGER, discharges INTEGER, general_bed_occupancy REAL, icu_occupancy REAL, doctors_available INTEGER, nurses_available INTEGER, diagnostic_utilization REAL)")
        conn.execute("CREATE TABLE IF NOT EXISTS capacities (resource TEXT PRIMARY KEY, capacity REAL NOT NULL)")
        conn.executemany("INSERT OR REPLACE INTO operational_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", df.itertuples(index=False, name=None))
        conn.executemany("INSERT OR REPLACE INTO capacities VALUES (?,?)", CAPACITIES.items())


def get_df() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM operational_records ORDER BY date", conn, parse_dates=["date"])


def get_configured_capacities() -> dict[str, int]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT resource, capacity FROM capacities").fetchall()
    configured = {resource: int(capacity) for resource, capacity in rows}
    if set(configured) != set(CAPACITIES):
        raise ValueError("Configured capacity table does not match modeled resources")
    return configured


@lru_cache(maxsize=1)
def forecast() -> dict[str, Any]:
    df = get_df()
    evaluation = evaluate_forecast_models(df)
    result = {k: [] for k in DEMAND_COLS}
    for col in DEMAND_COLS:
        work = make_features(df, col)
        x = work[list(FORECAST_FEATURES)]
        model = RandomForestRegressor(**MODEL_PARAMS).fit(x, work[col])
        history = df[col].astype(float).tolist()
        for step in range(1, 8):
            d = (df.date.max() + pd.Timedelta(days=step))
            row = pd.DataFrame([[d.dayofweek, d.month, history[-1], history[-7], float(np.mean(history[-7:]) if len(history) >= 7 else np.mean(history))]], columns=x.columns)
            val = max(0, float(model.predict(row)[0]))
            result[col].append(round(val, 1))
            history.append(val)
    days = [(df.date.max() + pd.Timedelta(days=i)).date().isoformat() for i in range(1, 8)]
    # Model requirements from forecast. Occupancy is the latest measured starting point.
    occ_general = float(df.general_bed_occupancy.iloc[-1])
    occ_icu = float(df.icu_occupancy.iloc[-1])
    capacities = get_configured_capacities()
    capacities["doctors"] = max(1, min(capacities["doctors"], int(df.doctors_available.iloc[-1])))
    capacities["nurses"] = max(1, min(capacities["nurses"], int(df.nurses_available.iloc[-1])))
    mean_emergency = float(df.emergency_admissions.tail(28).mean())
    mean_general = float(df.general_admissions.tail(28).mean())
    mean_icu = float(df.icu_admissions.tail(28).mean())
    mean_demand = float(df[DEMAND_COLS].tail(28).sum(axis=1).mean())
    daily = []
    for i, day in enumerate(days):
        demand = sum(result[c][i] for c in DEMAND_COLS)
        occ_general = float(np.clip(occ_general + (result["general_admissions"][i] - mean_general) * .30 + (result["emergency_admissions"][i] - mean_emergency) * .15 + (76 - occ_general) * .06, 0, 100))
        occ_icu = float(np.clip(occ_icu + (result["icu_admissions"][i] - mean_icu) * .35 + (15 - occ_icu) * .08, 0, 20))
        general_beds = occ_general
        icu_beds = occ_icu
        req = {"general_beds": general_beds, "icu": icu_beds,
               "doctors": max(0, demand * ASSUMPTIONS["doctors_per_admission"]),
               "nurses": max(0, (general_beds + icu_beds) * ASSUMPTIONS["nurses_per_occupied_bed"]),
               "diagnostics": float(np.clip(df.diagnostic_utilization.iloc[-1] + (demand - mean_demand) * ASSUMPTIONS["diagnostics_points_per_admission"], 0, 100))}
        reported_requirements = {key: round(value, 1) for key, value in req.items()}
        pressure = {key: round(value / capacities[key] * 100, 1) for key, value in reported_requirements.items()}
        daily.append({"date": day, "emergency_admissions": result[DEMAND_COLS[0]][i], "general_admissions": result[DEMAND_COLS[1]][i], "icu_admissions": result[DEMAND_COLS[2]][i], "requirements": reported_requirements, "pressure": pressure})
    return {"next_day": daily[0], "seven_day": daily, "model": "RandomForestRegressor with lag and calendar features", "evaluation": evaluation, "note": "Forecasts and holdout metrics use synthetic demonstration data and are not validated for real-world operations."}


def status(pct: float) -> str:
    return next(label for threshold, label in THRESHOLDS if pct >= threshold)


def get_operational_state(adjust: dict[str, float] | None = None) -> dict[str, Any]:
    is_scenario = adjust is not None
    adjust = adjust or {}
    fc = forecast()
    df = get_df()
    latest = df.iloc[-1]
    day = fc["next_day"]
    current_admissions = float(latest.emergency_admissions + latest.general_admissions + latest.icu_admissions)
    current = {"general_beds": float(latest.general_bed_occupancy), "icu": float(latest.icu_occupancy),
               "doctors": current_admissions * ASSUMPTIONS["doctors_per_admission"],
               "nurses": (float(latest.general_bed_occupancy) + float(latest.icu_occupancy)) * ASSUMPTIONS["nurses_per_occupied_bed"],
               "diagnostics": float(latest.diagnostic_utilization)}
    capacities = get_configured_capacities()
    capacities["nurses"] = max(1, min(CAPACITIES["nurses"], int(latest.nurses_available)))
    capacities["doctors"] = max(1, min(CAPACITIES["doctors"], int(latest.doctors_available)))
    baseline_capacities = dict(capacities)
    capacities["nurses"] = max(1, round(capacities["nurses"] * (1 + adjust.get("nurse_change_percent", 0) / 100)))
    capacities["doctors"] = max(1, round(capacities["doctors"] * (1 + adjust.get("doctor_change_percent", 0) / 100)))
    capacities["icu"] = max(1, capacities["icu"] + int(adjust.get("icu_capacity_change", 0)))
    ed = day["emergency_admissions"] * (1 + adjust.get("emergency_change_percent", 0) / 100)
    gd = day["general_admissions"] * (1 + adjust.get("general_admission_change_percent", 0) / 100)
    icud = day["icu_admissions"] * (1 + adjust.get("emergency_change_percent", 0) / 100)
    demand_total = ed + gd + icud
    baseline_demand = sum(day[k] for k in DEMAND_COLS)
    forecast_general_beds = day["requirements"]["general_beds"]
    forecast_icu_beds = day["requirements"]["icu"]
    projected_general_beds = max(0, forecast_general_beds
        + (gd - day["general_admissions"]) * ASSUMPTIONS["beds_per_general_admission"]
        + (ed - day["emergency_admissions"]) * ASSUMPTIONS["beds_per_emergency_admission"])
    projected_icu_beds = max(0, forecast_icu_beds
        + (icud - day["icu_admissions"]) * ASSUMPTIONS["icu_beds_per_icu_admission"])
    projected = {"general_beds": projected_general_beds,
                 "icu": projected_icu_beds,
                 "doctors": demand_total * ASSUMPTIONS["doctors_per_admission"],
                 "nurses": (projected_general_beds + projected_icu_beds) * ASSUMPTIONS["nurses_per_occupied_bed"],
                 "diagnostics": max(0, day["requirements"]["diagnostics"] + (demand_total - baseline_demand) * ASSUMPTIONS["diagnostics_points_per_admission"])}
    names = {"general_beds": "General beds", "icu": "ICU", "doctors": "Doctors (FTE)", "nurses": "Nurses (FTE)", "diagnostics": "Diagnostics"}
    resources = []
    for key, name in names.items():
        cap = capacities[key]
        cur = current[key]
        pred = projected[key]
        # For staff, current and predicted are available/required respectively. For capacity-pressure comparisons use requirement / available capacity.
        current_value = round(cur, 1)
        predicted_value = round(pred, 1)
        current_pct = round(current_value / cap * 100, 1)
        predicted_pct = round(predicted_value / cap * 100, 1)
        resources.append({"key": key, "name": name, "current": current_value, "capacity": cap, "utilization": current_pct, "predicted_value": predicted_value, "predicted_utilization": predicted_pct, "remaining": round(cap - current_value, 1), "predicted_remaining": round(cap - predicted_value, 1), "status": status(current_pct), "predicted_status": status(predicted_pct), "exceeds_capacity": predicted_pct > 100})
    ranked = sorted(resources, key=lambda x: x["predicted_utilization"], reverse=True)
    bottleneck = ranked[0]
    alerts = []
    for r in resources:
        p = r["predicted_utilization"]
        if p >= 100:
            severity, msg = "CRITICAL", f"{r['name']} demand is projected to exceed available capacity."
        elif p >= 85:
            severity, msg = "WARNING", f"{r['name']} capacity pressure is projected to reach {p:.0f}%."
        elif p >= 70:
            severity, msg = "WATCH", f"{r['name']} capacity pressure is projected to increase to {p:.0f}%."
        else:
            continue
        alerts.append({"severity": severity, "resource": r["name"], "message": msg, "current_value": r["utilization"], "predicted_value": p, "forecast_period": "Next operational day"})
    baseline_pressures = {key: {"value": round(day["requirements"][key], 1), "capacity": baseline_capacities[key], "projected_utilization": round(round(day["requirements"][key], 1) / baseline_capacities[key] * 100, 1), "current_utilization": round(round(current[key], 1) / baseline_capacities[key] * 100, 1)} for key in names}
    recs = recommendations(resources, day, adjust, baseline_pressures, is_scenario)
    projected_demand = {"emergency_admissions": round(ed, 1), "general_admissions": round(gd, 1), "icu_admissions": round(icud, 1), "total_admissions": round(demand_total, 1), "requirements": {key: round(value, 1) for key, value in projected.items()}}
    return {"resources": resources, "bottleneck": {"resource": bottleneck["name"], "key": bottleneck["key"], "predicted_utilization": bottleneck["predicted_utilization"], "exceeds_capacity": bottleneck["exceeds_capacity"], "ranked_pressures": [{"resource": r["name"], "utilization": r["predicted_utilization"], "status": r["predicted_status"]} for r in ranked]}, "alerts": alerts, "recommendations": recs, "forecast": fc, "projected_demand": projected_demand}


def recommendations(resources: list[dict], day: dict, adjust: dict, baseline_pressures: dict[str, dict], is_scenario: bool = False) -> list[dict]:
    by = {r["key"]: r for r in resources}
    out = []
    nurse = by["nurses"]
    if nurse["predicted_utilization"] >= PLANNING_UTILIZATION_THRESHOLD:
        safe_capacity = nurse["capacity"] * PLANNING_UTILIZATION_THRESHOLD / 100
        requirement = nurse["predicted_value"]
        gap = max(0, requirement - safe_capacity)
        qty = max(1, int(np.ceil(gap)))
        out.append({"action": f"Review reallocation of {qty} nurses to high-demand operations", "resource": "Nurses", "quantity": qty, "source": "Units with verified staffing headroom", "destination": "High-demand operations", "priority": "High" if nurse["predicted_utilization"] >= 95 else "Medium", "reason": f"Projected nursing requirement is {requirement:.0f} against {nurse['capacity']} available ({nurse['predicted_utilization']:.0f}% pressure).", "expected_effect": "Focuses existing coverage on the highest-pressure operations.", "why": [f"Available nurses: {nurse['capacity']}", f"Safe planning capacity at {PLANNING_UTILIZATION_THRESHOLD}% capacity pressure: {safe_capacity:.2f} nurses ({nurse['capacity']} × {PLANNING_UTILIZATION_THRESHOLD}%)", f"Projected nursing requirement: {requirement:.1f} nurses", f"Requirement above the safe planning threshold: {gap:.2f} nurses; ceil({gap:.2f}) = {qty} nurses to review for reallocation", "Reallocation is recommended because projected staffing pressure meets or exceeds the configured threshold; confirm donor-unit headroom before reallocating."]})
    for key, action in (("icu", "Review ICU reserve and transfer readiness"), ("general_beds", "Prepare configured overflow bed capacity"), ("doctors", "Review clinician coverage for the projected peak"), ("diagnostics", "Prioritize diagnostic slots for urgent workflows")):
        r = by[key]
        threshold = RECOMMENDATION_UTILIZATION_THRESHOLD
        if r["predicted_utilization"] >= threshold:
            baseline = baseline_pressures[key]
            is_explanation_resource = key in ("icu", "diagnostics")
            expected_effect = ("Highlights the ICU capacity constraint for reserve and transfer-readiness review." if key == "icu" else "Makes projected diagnostic pressure visible so slots can be prioritized for urgent workflows." if key == "diagnostics" else "Reduces risk of the projected constraint becoming an operational bottleneck.")
            why = [f"Current pressure: {baseline['current_utilization']:.1f}% ({r['current']:.1f} / {baseline['capacity']} configured capacity)", f"Baseline projected capacity pressure: {baseline['projected_utilization']:.1f}% ({baseline['value']:.1f} / {baseline['capacity']})"]
            if is_scenario:
                why.append(f"Scenario projected capacity pressure: {r['predicted_utilization']:.1f}% ({r['predicted_value']:.1f} / {r['capacity']} adjusted capacity)")
            elif not is_explanation_resource:
                why.append(f"Projected pressure: {r['predicted_utilization']:.1f}% ({r['predicted_value']:.1f} / {r['capacity']} configured capacity)")
            capacity_unit = "beds" if key == "icu" else "diagnostic points" if key == "diagnostics" else "units"
            capacity_detail = f"{r['capacity']} {capacity_unit}"
            if r["capacity"] != baseline["capacity"]:
                capacity_detail += f" (baseline: {baseline['capacity']})"
            why.extend([f"Configured capacity: {capacity_detail}", f"Recommendation threshold: {threshold}% projected capacity pressure", f"Reason: {'Scenario pressure' if is_scenario else 'Baseline projected capacity pressure'} is {r['predicted_utilization']:.1f}%, meeting or exceeding the configured {threshold}% threshold.", f"Expected effect: {expected_effect}"])
            out.append({"action": action, "resource": r["name"], "quantity": max(1, int(np.ceil(r["predicted_value"] - r["capacity"] * threshold / 100))), "source": "Available operational reserve", "destination": r["name"], "priority": "High" if r["predicted_utilization"] >= 95 else "Medium", "reason": f"Projected {r['name']} pressure is {r['predicted_utilization']:.0f}% of configured capacity.", "expected_effect": expected_effect, "why": why})
    return out[:4]


class SimulationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emergency_change_percent: float = Field(0, ge=-30, le=50)
    general_admission_change_percent: float = Field(0, ge=-30, le=50)
    nurse_change_percent: float = Field(0, ge=-30, le=30)
    doctor_change_percent: float = Field(0, ge=-30, le=30)
    icu_capacity_change: int = Field(0, ge=-5, le=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_db()
    yield


app = FastAPI(title="HospitaLens AI API", version="1.0.0", lifespan=lifespan)
allowed_origins = [origin.strip().rstrip("/") for origin in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/api/health")
@app.get("/health")
def health():
    return {"status": "ok", "service": "HospitaLens AI", "data_mode": "synthetic"}


@app.get("/api/dashboard")
def dashboard():
    state = get_operational_state()
    state["generated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    return state


@app.get("/api/resources")
def resources():
    state = get_operational_state()
    return {"resources": state["resources"], "bottleneck": state["bottleneck"]}


@app.get("/api/forecast")
def get_forecast():
    return forecast()


@app.get("/api/alerts")
def alerts():
    return {"alerts": get_operational_state()["alerts"]}


@app.get("/api/recommendations")
def get_recommendations():
    return {"recommendations": get_operational_state()["recommendations"]}


@app.post("/api/simulation")
def simulation(payload: SimulationInput):
    baseline = get_operational_state()
    changes = payload.model_dump()
    scenario = get_operational_state(changes)
    comparison = []
    base_by = {r["key"]: r for r in baseline["resources"]}
    for r in scenario["resources"]:
        b = base_by[r["key"]]
        comparison.append({"key": r["key"], "resource": r["name"], "current_utilization": b["utilization"], "baseline_forecast_utilization": b["predicted_utilization"], "scenario_utilization": r["predicted_utilization"], "change": round(r["predicted_utilization"] - b["predicted_utilization"], 1), "status": r["predicted_status"], "current_value": b["current"], "baseline_forecast_value": b["predicted_value"], "scenario_value": r["predicted_value"], "current_capacity": b["capacity"], "scenario_capacity": r["capacity"]})
    return {"baseline": baseline["resources"], "simulated": scenario["resources"], "comparison": comparison, "predicted_demand": scenario["projected_demand"], "bottleneck": scenario["bottleneck"], "alerts": scenario["alerts"], "recommendations": scenario["recommendations"], "changes": changes, "explanation": f"The {changes['emergency_change_percent']:+g}% emergency and {changes['general_admission_change_percent']:+g}% general-admission changes adjust next-day demand. Those adjusted admissions flow through projected bed, ICU, clinician, nursing and diagnostic requirements before capacity pressure, bottlenecks, alerts and recommendations are recalculated. Change is shown against the baseline forecast; Current matches the dashboard/resource snapshot."}


@app.post("/api/emergency-surge")
def emergency_surge():
    return simulation(SimulationInput(emergency_change_percent=30))
