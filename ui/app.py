
import os
import sys
import math
import requests

# -------------------------------------------------
# Project path setup
# -------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st


# -------------------------------------------------
# Configuration
# -------------------------------------------------

API_URL = os.getenv(
    "PARMANA_API_URL",
    "http://127.0.0.1:8000/ask"
)

REQUEST_TIMEOUT = 120


# -------------------------------------------------
# Page configuration
# -------------------------------------------------

st.set_page_config(
    page_title="ParmanaAI | Evidence Verification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------
# Styling
# -------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #888888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-label {
        font-size: 1.1rem;
        font-weight: 650;
        margin-top: 1rem;
    }

    .small-muted {
        color: #888888;
        font-size: 0.85rem;
    }

    .verdict {
        font-weight: 700;
        font-size: 0.9rem;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def safe_score(value, default=0.0):
    """Convert a score to a finite float."""

    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return max(0.0, min(1.0, result))

    except (TypeError, ValueError):
        return default


def display_verdict(verdict):
    """Display verification outcomes with consistent styling."""

    verdict = str(verdict or "UNKNOWN").upper()

    if verdict in {
        "SUPPORTED",
        "NUMERICALLY_CONSISTENT",
    }:
        st.success(verdict)

    elif verdict in {
        "CONTRADICTED",
        "NUMERICAL_DISCREPANCY",
        "CONTEXT_MISMATCH",
        "CONFLICTING_EVIDENCE",
        "CONFLICTING_SIGNALS",
    }:
        st.error(verdict)

    elif verdict in {
        "INSUFFICIENT_EVIDENCE",
        "INCONCLUSIVE",
        "NO_EVIDENCE_PROVIDED",
        "NO_NUMERICAL_EVIDENCE",
    }:
        st.warning(verdict)

    else:
        st.info(verdict)


def render_nli_details(claim):
    """Render the original NLI verification signal."""

    signals = claim.get("verification_signals") or {}
    nli = signals.get("nli") or {}

    nli_verdict = claim.get(
        "nli_verdict",
        nli.get("verdict", claim.get("verdict", "UNKNOWN")),
    )

    st.markdown("**NLI verification**")
    display_verdict(nli_verdict)

    scores = (
        nli.get("scores")
        or claim.get("nli_scores")
        or {}
    )

    if scores:
        score_cols = st.columns(3)

        for col, key in zip(
            score_cols,
            ["entailment", "contradiction", "neutral"],
        ):
            value = safe_score(scores.get(key))

            with col:
                st.metric(
                    label=key.capitalize(),
                    value=f"{value:.3f}",
                )


def render_context_comparison(context):
    """Render clinical-context matching details."""

    if not context:
        st.info("No clinical-context comparison available.")
        return

    display_verdict(context.get("status", "UNKNOWN"))

    dimensions = context.get("dimension_results") or {}

    if dimensions:
        rows = []

        for dimension, result in dimensions.items():
            rows.append({
                "Dimension": dimension.replace("_", " ").title(),
                "Status": result.get("status", "unknown"),
                "Claim": ", ".join(
                    result.get("claim_values") or []
                ),
                "Evidence": ", ".join(
                    result.get("evidence_values") or []
                ),
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

    if context.get("mismatches"):
        st.warning(
            "Mismatched dimensions: "
            + ", ".join(context["mismatches"])
        )

    if context.get("unknown_dimensions"):
        st.info(
            "Insufficient context for: "
            + ", ".join(context["unknown_dimensions"])
        )

    st.caption(
        "Lexical context comparison does not establish clinical equivalence."
    )


def render_numerical_comparison(comparison):
    """Render an individual numerical comparison."""

    if not comparison:
        st.info("No individual numerical comparison available.")
        return

    st.write(
        "**Comparison status:**",
        comparison.get("status", "unknown"),
    )

    st.write(
        "**Reason:**",
        comparison.get("reason", "Not provided."),
    )

    cols = st.columns(3)

    with cols[0]:
        st.write("**Claim value**")
        st.write(
            comparison.get("claim_value", "N/A")
        )

    with cols[1]:
        st.write("**Evidence value**")
        st.write(
            comparison.get("evidence_value", "N/A")
        )

    with cols[2]:
        st.write("**Canonical unit**")
        st.write(
            comparison.get("canonical_unit", "N/A")
        )


def render_numerical_assessment(assessment):
    """Render aggregated numerical verification."""

    if not assessment:
        st.info("No numerical assessment was returned.")
        return

    numerical_verdict = assessment.get(
        "verdict",
        "NO_NUMERICAL_EVIDENCE",
    )

    display_verdict(numerical_verdict)

    st.write(
        assessment.get(
            "reason",
            "No numerical explanation provided.",
        )
    )

    # Aggregate statistics

    metric_cols = st.columns(4)

    metrics = [
        ("Assessments", "source_assessment_count"),
        ("Documents", "unique_document_count"),
        ("Consistent", "consistent_source_count"),
        ("Discrepant", "discrepant_source_count"),
    ]

    for col, (label, key) in zip(metric_cols, metrics):
        with col:
            st.metric(
                label=label,
                value=assessment.get(key, 0),
            )

    additional_cols = st.columns(3)

    with additional_cols[0]:
        st.metric(
            "Context mismatches",
            assessment.get("context_mismatch_count", 0),
        )

    with additional_cols[1]:
        st.metric(
            "Inconclusive",
            assessment.get("inconclusive_source_count", 0),
        )

    with additional_cols[2]:
        st.metric(
            "Within-document conflicts",
            assessment.get("within_document_conflict_count", 0),
        )

    # Document-level outcomes

    document_results = assessment.get("document_results") or []

    if document_results:
        st.markdown("**Document-level outcomes**")

        st.dataframe(
            [
                {
                    "Document ID": item.get("document_id") or "Unknown",
                    "Verdict": item.get("verdict", "UNKNOWN"),
                    "Assessments": item.get("assessment_count", 0),
                }
                for item in document_results
            ],
            use_container_width=True,
            hide_index=True,
        )

    # Individual evidence assessments

    source_assessments = assessment.get(
        "source_assessments"
    ) or []

    if source_assessments:
        st.markdown("**Individual evidence assessments**")

        for index, source in enumerate(source_assessments, 1):
            source_assessment = source.get("assessment") or {}

            source_verdict = source_assessment.get(
                "verdict",
                "UNKNOWN",
            )

            label = (
                f"Evidence assessment {index} | "
                f"{source.get('document_id') or 'Unknown document'} | "
                f"{source_verdict}"
            )

            with st.expander(label):
                st.write(
                    "**Document ID:**",
                    source.get("document_id", "Unavailable"),
                )

                st.write(
                    "**Chunk ID:**",
                    source.get("chunk_id", "Unavailable"),
                )

                st.write(
                    "**Source type:**",
                    source.get("source_type", "unknown"),
                )

                st.write(
                    "**NLI verdict:**",
                    source.get("nli_verdict", "UNKNOWN"),
                )

                st.markdown("**Numerical assessment**")

                display_verdict(source_verdict)

                st.write(
                    source_assessment.get(
                        "reason",
                        "No reason provided.",
                    )
                )

                render_numerical_comparison(
                    source_assessment.get(
                        "numerical_comparison"
                    )
                )

                with st.expander("Clinical context details"):
                    render_context_comparison(
                        source_assessment.get(
                            "context_comparison"
                        )
                    )

    st.caption(
        "Numerical consistency and discrepancy are not independently "
        "medical truth or medical contradiction."
    )


def render_claim(claim, index):
    """Render one atomic claim and all available verification details."""

    claim_text = claim.get(
        "claim_text",
        claim.get("claim", "Claim text unavailable."),
    )

    combined_verdict = claim.get(
        "combined_verdict",
        claim.get("verdict", "UNKNOWN"),
    )

    nli_verdict = claim.get(
        "nli_verdict",
        claim.get("verdict", "UNKNOWN"),
    )

    with st.expander(
        f"Claim {index}: {claim_text}",
        expanded=(index == 1),
    ):
        st.markdown("#### Verification outcomes")

        left, right = st.columns(2)

        with left:
            st.markdown("**Original NLI verdict**")
            display_verdict(nli_verdict)

        with right:
            st.markdown("**Combined verdict**")
            display_verdict(combined_verdict)

        st.markdown("---")

        render_nli_details(claim)

        st.markdown("---")

        st.markdown("#### Numerical verification")

        render_numerical_assessment(
            claim.get("numerical_assessment")
        )

        # Supporting source and citation

        st.markdown("---")
        st.markdown("#### Supporting evidence")

        st.write(
            "**Supporting source type:**",
            claim.get("supporting_source", "Unknown"),
        )

        st.write(
            "**Citation match:**",
            claim.get("citation_match", "Unknown"),
        )

        citation = claim.get("citation")

        if citation:
            with st.expander("View citation text"):
                st.write(citation)

        st.markdown("---")

        st.markdown("#### Claim extraction metadata")

        metadata = {
            "Claim ID": claim.get("claim_id"),
            "Extraction type": claim.get("extraction_type"),
            "Inferential": claim.get("is_inferential"),
            "Definitive": claim.get("is_definitive"),
            "Causal language": claim.get("contains_causal_language"),
            "Atomic": claim.get("atomic"),
            "Numerical information": claim.get(
                "contains_numerical_information"
            ),
        }

        st.json(metadata)

        if claim.get("numerical_mentions"):
            with st.expander("Extracted numerical mentions"):
                st.json(claim["numerical_mentions"])

        if claim.get("explanation"):
            st.markdown("**Verification explanation**")
            st.write(claim["explanation"])


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    st.write(
        "Ask a question and inspect how ParmanaAI "
        "evaluates its claims against available evidence."
    )

    st.markdown("---")

    st.markdown("**Backend URL**")
    st.code(API_URL)

    st.markdown("**Request timeout**")
    st.write(f"{REQUEST_TIMEOUT} seconds")

    st.markdown("---")

    st.markdown("### Verification signals")

    st.caption(
        "NLI: Natural Language Inference"
    )

    st.caption(
        "Numerical: Measurement and value comparison"
    )

    st.caption(
        "Context: Explicit clinical-context matching"
    )

    st.markdown("---")

    st.caption(
        "ParmanaAI is an evidence-analysis prototype, "
        "not a clinically validated diagnostic system."
    )


# -------------------------------------------------
# Main header
# -------------------------------------------------

st.markdown(
    '<div class="main-header">🧠 ParmanaAI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Evidence-grounded medical claim verification and '
    'numerical consistency analysis'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# -------------------------------------------------
# Input section
# -------------------------------------------------

st.subheader("Ask a medical question")

with st.form("medical_query_form"):

    question = st.text_area(
        "Question",
        placeholder=(
            "Example: What evidence exists regarding "
            "the effectiveness and side effects of metformin?"
        ),
        height=120,
    )

    with st.expander("Advanced: Provide external evidence context"):
        external_context = st.text_area(
            "External context",
            placeholder=(
                "Optional: Paste research abstracts, "
                "guidelines, or other evidence text."
            ),
            height=180,
        )

        st.caption(
            "Externally supplied context is treated as unverified "
            "unless independently validated."
        )

    evaluate = st.form_submit_button(
        "🔍 Evaluate claims",
        type="primary",
        use_container_width=True,
    )


# -------------------------------------------------
# API request
# -------------------------------------------------

if evaluate:

    if not question.strip():
        st.warning("Please enter a question before evaluating.")

    else:

        request_payload = {
            "query": question.strip(),
        }

        if external_context.strip():
            request_payload["external_context"] = (
                external_context.strip()
            )

        with st.spinner(
            "Running ParmanaAI evidence and verification pipeline..."
        ):

            try:
                response = requests.post(
                    API_URL,
                    json=request_payload,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code != 200:
                    st.error(
                        f"Backend returned HTTP {response.status_code}"
                    )

                    try:
                        error_data = response.json()
                        st.json(error_data)
                    except ValueError:
                        st.code(response.text)

                    st.stop()

                data = response.json()

            except requests.exceptions.Timeout:
                st.error(
                    "The backend request timed out. "
                    "Try again or increase the configured timeout."
                )
                st.stop()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to ParmanaAI backend. "
                    "Make sure the FastAPI server is running."
                )
                st.stop()

            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

            except ValueError:
                st.error("The backend returned invalid JSON.")
                st.stop()

        if not isinstance(data, dict):
            st.error("Invalid response structure from backend.")
            st.stop()

        # -------------------------------------------------
        # Refusal and empty evidence handling
        # -------------------------------------------------

        if data.get("safe_refusal"):
            st.warning(
                data.get(
                    "message",
                    "Unable to answer safely.",
                )
            )
            st.stop()

        answer = data.get("answer", "")
        claims = data.get("claims") or []

        if not claims and data.get("hallucination_score", 0) >= 1:
            st.warning(
                "No verified medical evidence was found. "
                "The system did not produce verified claims."
            )

        # -------------------------------------------------
        # Answer
        # -------------------------------------------------

        st.divider()

        st.subheader("📋 Generated answer")

        st.markdown(answer or "No answer was generated.")

        # -------------------------------------------------
        # Overall metrics
        # -------------------------------------------------

        st.divider()

        st.subheader("📊 Evaluation overview")

        hallucination_score = safe_score(
            data.get("hallucination_score")
        )

        relevance_score = safe_score(
            data.get("relevance_score")
        )

        metric_cols = st.columns(3)

        with metric_cols[0]:
            st.metric(
                "Hallucination heuristic",
                f"{hallucination_score:.3f}",
            )

        with metric_cols[1]:
            st.metric(
                "Answer relevance",
                f"{relevance_score:.3f}",
            )

        with metric_cols[2]:
            st.metric(
                "Extracted claims",
                len(claims),
            )

        st.caption(
            "Lower hallucination heuristic scores indicate fewer "
            "penalties under the project's current scoring rules. "
            "These scores are not calibrated clinical probabilities."
        )

        with st.expander("How the evaluation scores work"):
            st.write(
                "The hallucination heuristic aggregates penalties "
                "from claim-level verification outcomes."
            )

            st.write(
                "The relevance score is generated by the existing "
                "generation component."
            )

            st.write(
                "Neither metric establishes medical accuracy, "
                "clinical safety, or suitability for treatment decisions."
            )

        # -------------------------------------------------
        # Claim-level verification
        # -------------------------------------------------

        st.divider()

        st.subheader("🔬 Claim-level verification")

        if claims:

            for index, claim in enumerate(claims, 1):
                render_claim(claim, index)

        else:
            st.info(
                "No atomic claims were extracted from the generated answer."
            )

        # -------------------------------------------------
        # Completion
        # -------------------------------------------------

        st.divider()

        st.caption(
            "ParmanaAI | Evidence-aware medical claim verification"
        )
