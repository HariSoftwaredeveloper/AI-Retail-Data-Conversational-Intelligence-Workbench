"""FastAPI REST Service for Retail Intelligence Workbench.
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import uuid
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.config import DATA_DIR, ARTIFACTS_DIR, BASE_DIR, EVAL_DIR
from app.core.security import sanitize_filename, check_for_prompt_injection
from app.profiling.profiler import profile_dataframe
from app.profiling.issue_detector import detect_issues
from app.cleaning.pipeline import run_pipeline
from app.retail.normalizer import enrich_products_dataset
from app.retail.metrics import compute_sales_metrics, compute_inventory_metrics, compute_category_analytics
from app.retail.scoring import score_product_health, score_customer_rfm
from app.assistant.planner import QueryPlanner
from app.assistant.validator import validate_plan
from app.assistant.executor import execute_plan
from app.assistant.explainer import format_grounded_answer
from app.assistant.session import SessionManager
from app.core.models import ChatTurn


app = FastAPI(
    title="AI Retail Data & Conversational Intelligence Workbench",
    version="1.0.0",
    description="Grounded, deterministic retail data cleaning and conversational analytics API."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dataset store in memory for workbench session
LOADED_DATASETS: Dict[str, pd.DataFrame] = {}
session_manager = SessionManager()
planner = QueryPlanner()


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default_session"
    dataset: Optional[str] = None


@app.on_event("startup")
def startup_load_sample_data():
    """Auto-load any existing CSV files from data/ directory."""
    for csv_file in DATA_DIR.glob("*.csv"):
        try:
            name = csv_file.stem
            df = pd.read_csv(csv_file)
            LOADED_DATASETS[name] = df
            # If not yet cleaned, run baseline clean
            clean_pipe = run_pipeline(df, name, f"init_{name}")
            LOADED_DATASETS[f"{name}_clean"] = clean_pipe["df_clean"]
            LOADED_DATASETS[f"{name}_raw"] = df
        except Exception:
            pass


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "datasets_loaded": list(LOADED_DATASETS.keys()),
        "version": "1.0.0",
    }


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset, persist raw file, and register in memory."""
    safe_name = sanitize_filename(file.filename)
    dest_path = DATA_DIR / safe_name
    content = await file.read()

    with open(dest_path, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(dest_path)
        base_name = dest_path.stem
        LOADED_DATASETS[base_name] = df
        LOADED_DATASETS[f"{base_name}_raw"] = df
        return {
            "status": "uploaded",
            "dataset_name": base_name,
            "rows": len(df),
            "columns": list(df.columns),
            "path": str(dest_path),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")


@app.get("/api/datasets")
def list_datasets():
    """List all currently active datasets."""
    datasets_info = []
    for name, df in LOADED_DATASETS.items():
        datasets_info.append({
            "name": name,
            "rows": len(df),
            "columns": list(df.columns),
            "is_clean": name.endswith("_clean"),
            "is_raw": name.endswith("_raw"),
        })
    return {"datasets": datasets_info}


def _sanitize_records(df: pd.DataFrame, limit: int = 25) -> List[Dict[str, Any]]:
    """Convert dataframe slice to JSON-compliant records with NaN/inf handling."""
    records = df.head(limit).to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None
            elif isinstance(v, (float, np.floating)):
                r[k] = round(float(v), 2)
            elif isinstance(v, (int, np.integer)):
                r[k] = int(v)
            elif not isinstance(v, (str, bool, int, float)):
                r[k] = str(v)
    return records


@app.post("/api/pipeline/profile")
def profile_dataset_endpoint(dataset_name: str = Form(...)):
    """Generate profile and detect issues for a dataset."""
    if dataset_name not in LOADED_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

    df = LOADED_DATASETS[dataset_name]
    profile = profile_dataframe(df, dataset_name)
    issues = detect_issues(df, dataset_name)
    return {
        "profile": profile.model_dump(),
        "issues": [iss.model_dump() for iss in issues],
        "preview": _sanitize_records(df, 20),
    }


@app.post("/api/pipeline/clean")
def clean_dataset_endpoint(dataset_name: str = Form(...)):
    """Run full cleaning pipeline on dataset with idempotency verification."""
    if dataset_name not in LOADED_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

    df_raw = LOADED_DATASETS[dataset_name]
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    pipeline_res = run_pipeline(
        df_raw=df_raw,
        dataset_name=dataset_name,
        run_id=run_id,
    )

    clean_df = pipeline_res["df_clean"]
    clean_name = f"{dataset_name}_clean"
    LOADED_DATASETS[clean_name] = clean_df

    # Return structured metadata and row previews for artifacts
    resp = dict(pipeline_res)
    resp.pop("df_clean", None)
    resp["raw_preview"] = _sanitize_records(df_raw, 20)
    resp["clean_preview"] = _sanitize_records(clean_df, 20)
    return resp


@app.get("/api/evaluate/results")
@app.get("/results.json")
def get_evaluation_results():
    """Retrieve the latest batch evaluation results."""
    results_path = BASE_DIR / "results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation results found. Please run evaluation batch first.")
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/evaluate/run")
def run_evaluation_batch_endpoint():
    """Run the batch evaluation contract on cases.json and return updated results."""
    from app.evaluate import run_batch_evaluation
    cases_path = BASE_DIR / "cases.json"
    results_path = BASE_DIR / "results.json"
    if not cases_path.exists():
        raise HTTPException(status_code=404, detail="cases.json not found in root workspace.")

    results = run_batch_evaluation(
        input_file=str(cases_path),
        output_file=str(results_path),
        artifacts_dir=str(EVAL_DIR),
    )
    return {
        "status": "completed",
        "cases_count": len(results) if isinstance(results, list) else 1,
        "results": results,
    }


@app.get("/api/retail/metrics")
def get_retail_metrics():
    """Compute auditable retail sales and inventory KPIs."""
    orders_df = next((df for k, df in LOADED_DATASETS.items() if "order" in k.lower() and k.endswith("_clean")), None)
    if orders_df is None:
        orders_df = next((df for k, df in LOADED_DATASETS.items() if "order" in k.lower()), None)

    inv_df = next((df for k, df in LOADED_DATASETS.items() if "inventory" in k.lower() and k.endswith("_clean")), None)
    if inv_df is None:
        inv_df = next((df for k, df in LOADED_DATASETS.items() if "inventory" in k.lower()), None)

    sales_kpis = compute_sales_metrics(orders_df) if orders_df is not None else {"note": "Orders dataset not loaded"}
    inv_kpis = compute_inventory_metrics(inv_df) if inv_df is not None else {"note": "Inventory dataset not loaded"}

    # Category breakdown if orders available
    cat_analytics = compute_category_analytics(orders_df) if orders_df is not None else []

    return {
        "sales_kpis": sales_kpis,
        "inventory_kpis": inv_kpis,
        "category_analytics": cat_analytics,
    }


@app.get("/api/retail/scores")
def get_retail_scores():
    """Compute explainable Product Health and Customer RFM scores."""
    prod_df = next((df for k, df in LOADED_DATASETS.items() if "product" in k.lower() and k.endswith("_clean")), None)
    if prod_df is None:
        prod_df = next((df for k, df in LOADED_DATASETS.items() if "product" in k.lower()), None)

    orders_df = next((df for k, df in LOADED_DATASETS.items() if "order" in k.lower() and k.endswith("_clean")), None)
    if orders_df is None:
        orders_df = next((df for k, df in LOADED_DATASETS.items() if "order" in k.lower()), None)

    product_scores = []
    if prod_df is not None:
        for _, row in prod_df.head(10).iterrows():
            score_data = score_product_health(row.to_dict())
            product_scores.append(score_data)

    customer_rfm = score_customer_rfm(orders_df) if orders_df is not None else []

    return {
        "product_health_scores": product_scores,
        "customer_rfm_segments": customer_rfm[:15],
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """Conversational endpoint: plan -> validate -> execute -> verify evidence."""
    if not LOADED_DATASETS:
        raise HTTPException(status_code=400, detail="No datasets are currently loaded.")

    session_id = req.session_id or "default_session"
    session = session_manager.get_or_create_session(session_id)

    # Set active dataset if provided
    if req.dataset and req.dataset in LOADED_DATASETS:
        session.active_dataset = req.dataset

    # 1. Plan
    plan = planner.plan(req.question, LOADED_DATASETS, session)

    # 2. Validate
    is_valid, val_errors, sanitized_plan = validate_plan(plan, LOADED_DATASETS)

    evidence = None
    status = "ok"

    if not is_valid:
        status = "validation_error"
        answer = {
            "summary": f"Plan validation failed: {'; '.join(val_errors)}",
            "type": "error",
            "grounded": False,
        }
    elif sanitized_plan.intent in ("refuse", "clarify"):
        status = sanitized_plan.intent
        answer = format_grounded_answer(sanitized_plan, None, req.question)
    else:
        # 3. Execute
        evidence = execute_plan(sanitized_plan, LOADED_DATASETS)
        answer = format_grounded_answer(sanitized_plan, evidence, req.question)

    trace = {
        "validation_errors": val_errors,
        "operations": evidence.operations if evidence else [],
        "rows_considered": evidence.row_count_considered if evidence else 0,
    }

    turn = ChatTurn(
        question=req.question,
        plan=sanitized_plan.model_dump(),
        answer=answer,
        trace=trace,
        status=status,
    )

    # Persist session state
    session_manager.append_turn(
        session_id=session_id,
        turn=turn,
        new_active_dataset=sanitized_plan.dataset,
        new_filters=sanitized_plan.filters if sanitized_plan.filters else None,
    )

    return {
        "session_id": session_id,
        "question": req.question,
        "plan": sanitized_plan.model_dump(),
        "answer": answer,
        "trace": trace,
        "evidence": evidence.model_dump() if evidence else None,
        "status": status,
    }


@app.get("/api/chat/history/{session_id}")
def get_chat_history(session_id: str):
    """Retrieve full persistent audit trail of questions, plans, execution, and answers."""
    session = session_manager.get_or_create_session(session_id)
    return {
        "session_id": session.session_id,
        "active_dataset": session.active_dataset,
        "turns": [turn.model_dump() for turn in session.history],
    }


# Mount Frontend Static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
