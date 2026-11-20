import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import cached, TTLCache

# ==============================================================================
# 1. APP SETUP & MODEL LOADING
# ==============================================================================
app = FastAPI(title="Loan Decision Support API", version="1.0")

# Enable CORS so your frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths to the models folder (assuming backend/ is next to models/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")

print("Loading Machine Learning Artifacts into memory...")
# Stage 1 Artifacts
m1_model = joblib.load(os.path.join(MODELS_DIR, 'stage1_approval_model.joblib'))
m1_scaler = joblib.load(os.path.join(MODELS_DIR, 'stage1_scaler.joblib'))
m1_features = joblib.load(os.path.join(MODELS_DIR, 'stage1_features.joblib'))
m1_test_data = joblib.load(os.path.join(MODELS_DIR, 'stage1_test_data.joblib'))

# Stage 2 Artifacts
m2_model = joblib.load(os.path.join(MODELS_DIR, 'stage2_repayment_model.joblib'))
m2_scaler = joblib.load(os.path.join(MODELS_DIR, 'stage2_scaler.joblib'))
m2_features = joblib.load(os.path.join(MODELS_DIR, 'stage2_features.joblib'))
m2_test_data = joblib.load(os.path.join(MODELS_DIR, 'stage2_test_data.joblib'))
print("Artifacts loaded successfully.")

# Setup TTL Cache (Cache results for 5 minutes / 300 seconds to ensure <5ms latency on sliders)
calc_cache = TTLCache(maxsize=100, ttl=300)

# ==============================================================================
# 2. PYDANTIC SCHEMAS (Input Validation)
# ==============================================================================
class Stage1Request(BaseModel):
    loan_amnt: float
    fico_range_low: float
    dti: float
    emp_length: float
    addr_state: str

class Stage2Request(BaseModel):
    loan_amnt: float
    term: int
    int_rate: float
    installment: float
    grade: str
    emp_length: float
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    dti: float
    fico_range_low: float
    open_acc: float
    revol_bal: float
    revol_util: float

# ==============================================================================
# 3. PREDICTION ENDPOINTS
# ==============================================================================
@app.post("/api/predict/stage1")
def predict_eligibility(data: Stage1Request):
    """Predicts Historical Loan Approval (Stage 1)"""
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Feature Engineering: dti_to_fico
        df['dti_to_fico'] = df['dti'] / (df['fico_range_low'] + 1)
        
        # Handle state grouping (match training logic)
        top_states = ['CA', 'NY', 'TX', 'FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA']
        df['addr_state_grouped'] = np.where(df['addr_state'].isin(top_states), df['addr_state'], 'OTHER')
        df = df.drop(columns=['addr_state'])
        
        # One-hot encode and align columns with training features
        df_encoded = pd.get_dummies(df, columns=['addr_state_grouped'])
        df_encoded = df_encoded.reindex(columns=m1_features, fill_value=0)
        
        # Scale and predict
        X_scaled = m1_scaler.transform(df_encoded)
        prob = m1_model.predict_proba(X_scaled)[0][1]
        
        return {
            "status": "success",
            "approval_probability": round(float(prob), 4),
            "decision": "Approved" if prob >= 0.5 else "Rejected"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict/stage2")
def predict_repayment_risk(data: Stage2Request):
    """Predicts Future Repayment Risk for Approved Loans (Stage 2)"""
    try:
        df = pd.DataFrame([data.dict()])
        
        # Feature Engineering
        df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
        df['installment_to_monthly_inc'] = df['installment'] / ((df['annual_inc'] / 12.0) + 1)
        
        # One-hot encode and align columns
        df_encoded = pd.get_dummies(df, columns=['grade', 'home_ownership', 'verification_status', 'purpose'])
        df_encoded = df_encoded.reindex(columns=m2_features, fill_value=0)
        
        # Scale and predict
        X_scaled = m2_scaler.transform(df_encoded)
        prob = m2_model.predict_proba(X_scaled)[0][1]
        
        return {
            "status": "success",
            "default_probability": round(float(prob), 4),
            "risk_assessment": "High Risk (Default)" if prob >= 0.5 else "Low Risk (Will Pay)"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==============================================================================
# 4. ADVANCED ANALYTICS ENDPOINTS (The Differentiators)
# ==============================================================================
@app.get("/api/optimize-threshold")
@cached(cache=calc_cache)
def optimize_cost_matrix(cost_false_approval: float = 5.0, cost_false_rejection: float = 1.0):
    """Dynamically sweeps thresholds to find the minimum financial cost."""
    X_test = m1_test_data['X_test_scaled']
    y_test = m1_test_data['y_test']
    
    # Get probabilities for the entire test set
    probs = m1_model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.5
    min_cost = float('inf')
    cost_curve = []
    
    # Sweep thresholds from 0.05 to 0.95
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (probs >= t).astype(int)
        
        # False Positive (False Approval) and False Negative (False Rejection)
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))
        
        total_cost = (fp * cost_false_approval) + (fn * cost_false_rejection)
        cost_curve.append({"threshold": round(t, 2), "total_cost": float(total_cost)})
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = t
            
    return {
        "optimal_threshold": round(best_threshold, 2),
        "minimum_portfolio_cost": float(min_cost),
        "cost_curve": cost_curve
    }

@app.get("/api/fairness-audit")
@cached(cache=calc_cache)
def subgroup_fairness_audit():
    """Audits the Stage 2 model for error disparities across demographic groups."""
    X_test = m2_test_data['X_test_scaled']
    y_test = m2_test_data['y_test']
    groups = m2_test_data['groups']
    
    preds = m2_model.predict(X_test)
    
    # Merge predictions back with the demographic groups
    results = groups.copy()
    results['actual_risk'] = y_test
    results['predicted_risk'] = preds
    
    # Calculate False Positive Rate (Unfairly flagged as risk) per Home Ownership group
    audit_report = []
    for home_type in results['home_ownership'].unique():
        subset = results[results['home_ownership'] == home_type]
        if len(subset) == 0: continue
            
        actual_good = subset[subset['actual_risk'] == 0]
        fpr = 0 if len(actual_good) == 0 else np.sum(actual_good['predicted_risk'] == 1) / len(actual_good)
        
        audit_report.append({
            "home_ownership": str(home_type),
            "total_applicants": len(subset),
            "false_positive_rate": round(float(fpr), 4)
        })
        
    return {"status": "success", "fairness_audit_by_housing": audit_report}