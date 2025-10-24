import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from .healthy_comparator import healthy_sbp_for_age_and_sex


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Profile:
    sex: str  # 'male' or 'female'
    sbp: float  # mmHg
    anti_htn_meds: bool  # on antihypertensives
    non_hdl_c: float  # mg/dL
    hdl_c: float  # mg/dL
    statin: bool
    t2dm: bool  # diabetes present
    smoking: bool  # current smoker (past 30 days)
    egfr: float  # mL/min/1.73m^2
    # Optional PREVENT add-ons (use if your model includes them)
    hba1c: Optional[float] = None  # %
    uacr: Optional[float] = None  # mg/g
    sdi: Optional[float] = None  # Social Deprivation Index (zip-derived)
    # You can also carry zip if your code maps SDI from it


Age = float
RiskFn = Callable[[Age, Profile, Dict], float]  # returns probability in [0,1]


# -----------------------------
# Reference comparators
# -----------------------------
def optimal_reference_profile(age: Age, sex: str) -> Profile:
    """
    PREVENT risk age definition uses 'optimal' risk factor levels.
    Choose values that reflect guideline-optimal ranges and the PREVENT calculator inputs.
    Adjust if your team adopts slightly different 'optimal' anchors.
    """
    # Using optimal HDL from MESA heart age calculator: https://pubmed.ncbi.nlm.nih.gov/33663219/
    # Women need higher HDL: https://my.clevelandclinic.org/health/articles/11920-cholesterol-numbers-what-do-they-mean
    # Optimal non-hdl cholesterol is 120 mg/dL: https://pubmed.ncbi.nlm.nih.gov/40737004/, generally recommended to be below 130 mg/dL (https://pubmed.ncbi.nlm.nih.gov/32562186/)

    optimal_hdl_c = 55.0 if sex == "female" else 45.0
    sbp_from_lookup = healthy_sbp_for_age_and_sex(age, sex)
    return Profile(
        sex=sex,
        sbp=sbp_from_lookup,  # derived from comparator lookup table, interpolated by age
        anti_htn_meds=False,
        non_hdl_c=120.0,  # mg/dL, desirable non-hdl cholesterol
        hdl_c=optimal_hdl_c,
        statin=False,
        t2dm=False,
        smoking=False,
        egfr=90.0,  # normal kidney function
        hba1c=None,  # leave None unless using add-on model
        uacr=None,
        sdi=None,
    )


# -----------------------------
# Heart-age solver
# -----------------------------
class NoSolutionWarning(UserWarning):
    pass


def _bisect_for_age(
    target_risk: float,
    ref_profile_builder: Callable[[Age, str], Profile],
    sex: str,
    risk_fn: RiskFn,
    risk_kwargs: Dict,
    lo: float = 30.0,
    hi: float = 79.0,
    tol: float = 1e-4,
    max_iter: int = 1024,
) -> Tuple[Age, bool]:
    """Binary search for age* s.t. risk_fn(age*, ref_profile(age*, sex)) == target_risk."""
    # bracket
    r_lo = risk_fn(lo, ref_profile_builder(lo, sex), risk_kwargs)
    r_hi = risk_fn(hi, ref_profile_builder(hi, sex), risk_kwargs)

    if target_risk <= r_lo:
        # below lower bound — clamp to lo
        return lo, False
    if target_risk >= r_hi:
        # above upper bound — clamp to hi
        return hi, False

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid = risk_fn(mid, ref_profile_builder(mid, sex), risk_kwargs)
        if abs(r_mid - target_risk) < tol:
            return mid, True
        if r_mid < target_risk:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def prevent_heart_age(
    person_age: Age,
    person_profile: Profile,
    risk_fn: RiskFn,
    risk_kwargs: Dict,
    bounds: Tuple[Age, Age] = (30.0, 79.0),
    tol: float = 1e-4,
) -> Dict:
    """
    Compute PREVENT risk age via risk equivalence:
      1) Compute patient's absolute risk at their current age.
      2) Find age* for a reference comparator profile with the same predicted risk.

    Parameters
    ----------
    person_age : float
        Patient's chronological age (years).
    person_profile : Profile
        Patient's current predictors (excluding age).
    risk_fn : callable
        Your PREVENT risk function: risk_fn(age, profile, risk_kwargs)->prob.
        It should implement the PREVENT equation you chose (CVD/ASCVD/HF; 10y/30y; base or add-on).
    risk_kwargs : dict
        Extra args your risk_fn needs (e.g., which outcome/horizon/model variant, coefficients).
    bounds : (lo, hi)
        Valid age range; PREVENT validated for 30–79.
    tol : float
        Tolerance for the solver.

    Returns
    -------
    dict with fields: risk, heart_age, delta_years, solved_exactly(bool), clamped(bool)
    """

    lo, hi = bounds

    # 1) Patient's predicted risk at their age
    risk = risk_fn(person_age, person_profile, risk_kwargs)

    # 2) Solve for the equivalent age under reference profile
    heart_age, solved = _bisect_for_age(
        target_risk=risk,
        ref_profile_builder=optimal_reference_profile,
        sex=person_profile.sex,
        risk_fn=risk_fn,
        risk_kwargs=risk_kwargs,
        lo=lo,
        hi=hi,
        tol=tol,
    )
    clamped = (heart_age in (lo, hi)) and not solved

    return {
        "risk": risk,
        "heart_age": heart_age,
        "delta_years": heart_age - person_age,
        "solved_exactly": solved,
        "clamped": clamped,
    }


def prevent_ascvd_risk_10y_base(age: Age, p: Profile, kw: Dict) -> float:
    """
    PREVENT 10-year ASCVD risk - Base Model.
    Coefficients from Table S12.A. of the PREVENT paper supplement.
    """
    coeffs = {
        "female": {
            "age_10y": 0.719883, "non_hdl_c_1mmol": 0.1176967, "hdl_c_0_3mmol": -0.151185,
            "sbp_lt_110_20mmhg": -0.0835358, "sbp_ge_110_20mmhg": 0.3592852, "diabetes": 0.8348585,
            "smoking": 0.4831078, "egfr_lt_60_15ml": 0.4864619, "egfr_ge_60_15ml": 0.0397779,
            "anti_htn_meds": 0.2265309, "statin": -0.0592374, "treated_sbp_ge_110_20mmhg": -0.0395762,
            "treated_non_hdl_c": 0.0844423, "age_10y_x_non_hdl_c_1mmol": -0.0567839,
            "age_10y_x_hdl_c_1mmol": 0.0325692, "age_10y_x_sbp_ge_110_20mmhg": -0.1035985,
            "age_10y_x_diabetes": -0.2417542, "age_10y_x_smoking": -0.0791142,
            "age_10y_x_egfr_lt_60_15ml": -0.1671492, "constant": -3.819975
        },
        "male": {
            "age_10y": 0.7099847, "non_hdl_c_1mmol": 0.1658663, "hdl_c_0_3mmol": -0.1144285,
            "sbp_lt_110_20mmhg": -0.2837212, "sbp_ge_110_20mmhg": 0.3239977, "diabetes": 0.7189597,
            "smoking": 0.3956973, "egfr_lt_60_15ml": 0.3690075, "egfr_ge_60_15ml": 0.0203619,
            "anti_htn_meds": 0.2036522, "statin": -0.0865581, "treated_sbp_ge_110_20mmhg": -0.0322916,
            "treated_non_hdl_c": 0.114563, "age_10y_x_non_hdl_c_1mmol": -0.0300005,
            "age_10y_x_hdl_c_1mmol": 0.0232747, "age_10y_x_sbp_ge_110_20mmhg": -0.0927024,
            "age_10y_x_diabetes": -0.2018525, "age_10y_x_smoking": -0.0970527,
            "age_10y_x_egfr_lt_60_15ml": -0.1217081, "constant": -3.500655
        }
    }
    
    c = coeffs[p.sex]

    # Convert mg/dL to mmol/L for cholesterol
    mgdl_to_mmol = 38.67
    non_hdl_c_mmol = p.non_hdl_c / mgdl_to_mmol
    hdl_c_mmol = p.hdl_c / mgdl_to_mmol
    
    # Variable transformations (centered and scaled)
    age_term = (age - 55) / 10
    non_hdl_c_term = non_hdl_c_mmol - 3.5
    hdl_c_term = (hdl_c_mmol - 1.3) / 0.3
    sbp_lt_110_term = (min(p.sbp, 110) - 110) / 20
    sbp_ge_110_term = (max(p.sbp, 110) - 130) / 20
    egfr_lt_60_term = (min(p.egfr, 60) - 60) / -15
    egfr_ge_60_term = (max(p.egfr, 60) - 90) / -15 # Note: table indicates per -15ml for both

    t2dm_term = 1 if p.t2dm else 0
    smoking_term = 1 if p.smoking else 0
    statin_term = 1 if p.statin else 0
    anti_htn_meds_term = 1 if p.anti_htn_meds else 0
    
    # Linear predictor (log-odds)
    lp = c["constant"]
    lp += c["age_10y"] * age_term
    lp += c["non_hdl_c_1mmol"] * non_hdl_c_term
    lp += c["hdl_c_0_3mmol"] * hdl_c_term
    lp += c["sbp_lt_110_20mmhg"] * sbp_lt_110_term
    lp += c["sbp_ge_110_20mmhg"] * sbp_ge_110_term
    lp += c["diabetes"] * t2dm_term
    lp += c["smoking"] * smoking_term
    lp += c["egfr_lt_60_15ml"] * egfr_lt_60_term
    lp += c["egfr_ge_60_15ml"] * egfr_ge_60_term
    lp += c["statin"] * statin_term
    lp += c["anti_htn_meds"] * anti_htn_meds_term
    
    # Interaction terms
    lp += c["treated_sbp_ge_110_20mmhg"] * anti_htn_meds_term * sbp_ge_110_term
    lp += c["treated_non_hdl_c"] * statin_term * non_hdl_c_term
    lp += c["age_10y_x_non_hdl_c_1mmol"] * age_term * non_hdl_c_term
    lp += c["age_10y_x_hdl_c_1mmol"] * age_term * (hdl_c_mmol - 1.3) # Per 1mmol/L in table
    lp += c["age_10y_x_sbp_ge_110_20mmhg"] * age_term * sbp_ge_110_term
    lp += c["age_10y_x_diabetes"] * age_term * t2dm_term
    lp += c["age_10y_x_smoking"] * age_term * smoking_term
    lp += c["age_10y_x_egfr_lt_60_15ml"] * age_term * egfr_lt_60_term

    # BMI terms are omitted as coefficients are missing from the provided table.

    # Logistic approximation for risk
    return 1.0 / (1.0 + math.exp(-lp))
