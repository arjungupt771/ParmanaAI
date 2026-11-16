"""
ParmanaAI - Clinical Query Understanding and Expansion

Extracts clinical entities, expands abbreviations safely, and constructs
controlled expansions without altering the original medical intent.
"""

import re
from typing import Any, Dict, List, Set


# Standard medical abbreviations and their safe canonical expansions
MEDICAL_ABBREVIATIONS: Dict[str, str] = {
    "htn": "hypertension",
    "t2dm": "type 2 diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "dm1": "type 1 diabetes mellitus",
    "mi": "myocardial infarction",
    "bp": "blood pressure",
    "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure",
    "ckd": "chronic kidney disease",
    "copd": "chronic obstructive pulmonary disease",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "hba1c": "glycated hemoglobin",
    "a1c": "glycated hemoglobin",
    "ldl": "low density lipoprotein",
    "hdl": "high density lipoprotein",
    "cvd": "cardiovascular disease",
    "aki": "acute kidney injury",
    "bmi": "body mass index",
}

COMMON_CONDITIONS: Set[str] = {
    "hypertension", "high blood pressure", "diabetes", "type 2 diabetes",
    "type 1 diabetes", "myocardial infarction", "heart attack",
    "heart failure", "chronic kidney disease", "stroke", "asthma",
    "copd", "pneumonia", "fever", "hyperlipidemia", "coronary artery disease",
    "atrial fibrillation", "arrhythmia", "dyslipidemia", "nephropathy",
    "neuropathy", "retinopathy", "sepsis"
}

COMMON_MEDICATIONS: Set[str] = {
    "metformin", "lisinopril", "aspirin", "atorvastatin", "simvastatin",
    "amlodipine", "losartan", "insulin", "empagliflozin", "dapagliflozin",
    "metoprolol", "atenolol", "hydrochlorothiazide", "furosemide",
    "omeprazole", "paracetamol", "acetaminophen", "ibuprofen", "warfarin",
    "apixaban", "clopidogrel", "glipizide"
}

POPULATION_KEYWORDS: Dict[str, str] = {
    "child": "pediatric",
    "children": "pediatric",
    "pediatric": "pediatric",
    "infant": "infant",
    "neonate": "neonate",
    "adult": "adult",
    "adults": "adult",
    "elderly": "older_adult",
    "geriatric": "older_adult",
    "older adults": "older_adult",
    "pregnant": "pregnancy",
    "pregnancy": "pregnancy",
}

MEASUREMENT_KEYWORDS: Set[str] = {
    "blood pressure", "systolic blood pressure", "diastolic blood pressure",
    "blood glucose", "fasting blood glucose", "hba1c", "heart rate",
    "pulse", "ldl", "hdl", "cholesterol", "creatinine", "temperature",
    "body mass index", "bmi", "oxygen saturation", "spo2"
}


def extract_query_entities(query: str) -> Dict[str, List[str]]:
    """
    Extract structured medical entities from the query text.
    """
    lowered = query.lower()
    tokens = re.findall(r"\b[a-z0-9-]+\b", lowered)

    found_conditions = []
    for cond in sorted(COMMON_CONDITIONS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(cond) + r"\b", lowered):
            found_conditions.append(cond)

    found_medications = []
    for med in sorted(COMMON_MEDICATIONS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(med) + r"\b", lowered):
            found_medications.append(med)

    found_populations = []
    for term, canon in POPULATION_KEYWORDS.items():
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            if canon not in found_populations:
                found_populations.append(canon)

    found_measurements = []
    for meas in sorted(MEASUREMENT_KEYWORDS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(meas) + r"\b", lowered):
            found_measurements.append(meas)

    found_abbreviations = []
    for token in tokens:
        if token in MEDICAL_ABBREVIATIONS:
            found_abbreviations.append((token, MEDICAL_ABBREVIATIONS[token]))

    return {
        "conditions": found_conditions,
        "medications": found_medications,
        "populations": found_populations,
        "measurements": found_measurements,
        "abbreviations": found_abbreviations,
    }


def expand_query(query: str) -> Dict[str, Any]:
    """
    Produce query understanding metadata and controlled expansion.
    Always preserves original user query as the primary query.
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "original_query": "",
            "expanded_query": "",
            "entities": {},
            "sub_queries": [],
            "expansions": [],
        }

    original_query = query.strip()
    entities = extract_query_entities(original_query)

    # Controlled abbreviation expansion
    expansions = []
    for abbr, expanded in entities.get("abbreviations", []):
        expansions.append(expanded)

    # Detect multi-part medical questions (e.g. split on "and", "or", "vs")
    sub_queries = []
    conjunction_splits = re.split(r"\b(?:and|vs|versus)\b", original_query, flags=re.IGNORECASE)
    if len(conjunction_splits) > 1:
        for part in conjunction_splits:
            cleaned_part = part.strip(" ?.,;")
            if len(cleaned_part.split()) >= 3:
                sub_queries.append(cleaned_part)

    # Form expanded query string preserving original text first
    expansion_terms = [e for e in expansions if e.lower() not in original_query.lower()]
    if expansion_terms:
        expanded_query = f"{original_query} ({' '.join(expansion_terms)})"
    else:
        expanded_query = original_query

    return {
        "original_query": original_query,
        "expanded_query": expanded_query,
        "entities": entities,
        "sub_queries": sub_queries,
        "expansions": expansion_terms,
    }
