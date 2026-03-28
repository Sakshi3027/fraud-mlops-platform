import os
import json
import redis
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy import stats

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", 0.05))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", 100))

REFERENCE_STATS = {
    "amount": {"mean": 152.5, "std": 89.3},
    "transactions_last_hour": {"mean": 2.1, "std": 1.2},
    "is_international": {"mean": 0.10, "std": 0.30},
    "hour_of_day": {"mean": 15.2, "std": 4.8},
}


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True
    )


def fetch_recent_predictions(n: int = 500) -> pd.DataFrame:
    try:
        client = get_redis_client()
        raw = client.lrange("prediction_log", 0, n - 1)
        if not raw:
            return pd.DataFrame()
        records = [json.loads(r) for r in raw]
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Redis unavailable: {e}")
        return pd.DataFrame()


def detect_drift(df: pd.DataFrame) -> dict:
    if len(df) < MIN_SAMPLES:
        return {
            "drift_detected": False,
            "reason": f"insufficient samples ({len(df)} < {MIN_SAMPLES})",
        "recommendation": "INSUFFICIENT_DATA",
            "sample_count": len(df),
            "checks": {}
        }

    checks = {}
    drift_detected = False

    fraud_rate = df["prediction"].astype(int).mean() if "prediction" in df.columns else 0
    expected_fraud_rate = 0.02
    if fraud_rate > expected_fraud_rate * 5:
        drift_detected = True
        checks["fraud_rate"] = {
            "status": "DRIFT",
            "current": round(float(fraud_rate), 4),
            "expected": expected_fraud_rate,
            "reason": "fraud rate 5x above baseline"
        }
    else:
        checks["fraud_rate"] = {
            "status": "OK",
            "current": round(float(fraud_rate), 4),
            "expected": expected_fraud_rate
        }

    if "probability" in df.columns:
        probs = df["probability"].astype(float)
        high_uncertainty = ((probs > 0.3) & (probs < 0.7)).mean()
        if high_uncertainty > 0.3:
            drift_detected = True
            checks["prediction_uncertainty"] = {
                "status": "DRIFT",
                "high_uncertainty_rate": round(float(high_uncertainty), 4),
                "threshold": 0.3,
                "reason": "too many uncertain predictions"
            }
        else:
            checks["prediction_uncertainty"] = {
                "status": "OK",
                "high_uncertainty_rate": round(float(high_uncertainty), 4)
            }

    if "latency_ms" in df.columns:
        avg_latency = df["latency_ms"].astype(float).mean()
        if avg_latency > 100:
            checks["latency"] = {
                "status": "WARNING",
                "avg_latency_ms": round(float(avg_latency), 2),
                "threshold_ms": 100
            }
        else:
            checks["latency"] = {
                "status": "OK",
                "avg_latency_ms": round(float(avg_latency), 2)
            }

    champion_preds = df[df["model_role"] == "champion"] if "model_role" in df.columns else df
    challenger_preds = df[df["model_role"] == "challenger"] if "model_role" in df.columns else pd.DataFrame()

    if len(champion_preds) > 10 and len(challenger_preds) > 10:
        champ_fraud = champion_preds["prediction"].astype(int).mean()
        chall_fraud = challenger_preds["prediction"].astype(int).mean()
        divergence = abs(float(champ_fraud) - float(chall_fraud))
        if divergence > 0.15:
            drift_detected = True
            checks["model_divergence"] = {
                "status": "DRIFT",
                "champion_fraud_rate": round(float(champ_fraud), 4),
                "challenger_fraud_rate": round(float(chall_fraud), 4),
                "divergence": round(divergence, 4),
                "reason": "champion and challenger diverging significantly"
            }
        else:
            checks["model_divergence"] = {
                "status": "OK",
                "divergence": round(divergence, 4)
            }

    return {
        "drift_detected": drift_detected,
        "sample_count": len(df),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "checks": checks,
        "recommendation": "RETRAIN" if drift_detected else "OK"
    }


def trigger_retrain(drift_report: dict):
    print("\n DRIFT DETECTED — Triggering retrain pipeline...")
    print(f" Reason: {json.dumps(drift_report['checks'], indent=2)}")

    retrain_event = {
        "triggered_at": datetime.now(tz=timezone.utc).isoformat(),
        "trigger_reason": "drift_detected",
        "drift_report": drift_report
    }

    try:
        client = get_redis_client()
        client.lpush("retrain_events", json.dumps(retrain_event))
        client.ltrim("retrain_events", 0, 99)
        print(" Retrain event logged to Redis")
    except Exception as e:
        print(f" Redis unavailable, logging locally: {e}")

    retrain_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "scripts/retrain.sh"
    )
    if os.path.exists(retrain_script):
        print(f" Running retrain script: {retrain_script}")
        os.system(f"bash {retrain_script} &")
    else:
        print(" Retrain script not found — in production this triggers Kubeflow pipeline")

    return retrain_event


def run_drift_check():
    print(f"Running drift check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fetching recent predictions from Redis...")

    df = fetch_recent_predictions(500)

    if df.empty:
        print("No predictions found in Redis — generating synthetic data for demo...")
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "prediction": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "probability": np.random.beta(2, 8, n),
            "model_role": np.random.choice(
                ["champion", "challenger", "shadow"],
                n, p=[0.8, 0.15, 0.05]
            ),
            "latency_ms": np.random.exponential(15, n)
        })

    report = detect_drift(df)

    print(f"\n Drift Check Report")
    print(f" Samples analyzed: {report['sample_count']}")
    print(f" Drift detected:   {report['drift_detected']}")
    print(f" Recommendation:   {report['recommendation']}")
    print(f"\n Checks:")
    for check_name, result in report["checks"].items():
        status = result.get("status", "?")
        symbol = "✓" if status == "OK" else "!" if status == "WARNING" else "✗"
        print(f"   {symbol} {check_name}: {status}")

    if report["drift_detected"]:
        trigger_retrain(report)
    else:
        print("\n No drift detected — models are healthy")

    return report


if __name__ == "__main__":
    report = run_drift_check()
    print(f"\nFull report: {json.dumps(report, indent=2)}")