import {
  AlertTriangle,
  CheckCircle2,
  CircleHelp,
  FlaskConical,
  Info,
  XCircle,
} from 'lucide-react'

import PageHeader from '../components/ui/PageHeader'
import EmptyState from '../components/ui/EmptyState'
import { useAnalysisContext } from '../context/AnalysisContext'

import type {
  ClaimRecord,
  NumericalAssessment,
} from '../types/verification'

function formatVerdict(value?: string): string {
  if (!value) return 'Not available'

  return value
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

function getStatusClass(value?: string): string {
  switch (value?.toUpperCase()) {
    case 'NUMERICALLY_CONSISTENT':
      return 'verdict-supported'

    case 'NUMERICAL_DISCREPANCY':
      return 'verdict-contradicted'

    case 'CONTEXT_MISMATCH':
    case 'INCONCLUSIVE':
    case 'CONFLICTING_EVIDENCE':
    case 'CONFLICTING_SIGNALS':
    case 'NO_NUMERICAL_EVIDENCE':
      return 'verdict-caution'

    default:
      return 'verdict-neutral'
  }
}

function getStatusIcon(value?: string) {
  switch (value?.toUpperCase()) {
    case 'NUMERICALLY_CONSISTENT':
      return CheckCircle2

    case 'NUMERICAL_DISCREPANCY':
      return XCircle

    case 'CONTEXT_MISMATCH':
    case 'CONFLICTING_EVIDENCE':
    case 'CONFLICTING_SIGNALS':
      return AlertTriangle

    default:
      return CircleHelp
  }
}

function getClaimText(claim: ClaimRecord): string {
  return (
    claim.claim_text ||
    claim.claim ||
    claim.original_sentence ||
    'Claim text unavailable'
  )
}

function AssessmentCard({
  claim,
  index,
  assessment,
}: {
  claim: ClaimRecord
  index: number
  assessment: NumericalAssessment
}) {
  const StatusIcon = getStatusIcon(assessment.verdict)

  const metrics = [
    ['Source assessments', assessment.source_assessment_count],
    ['Unique documents', assessment.unique_document_count],
    ['Consistent sources', assessment.consistent_source_count],
    ['Discrepant sources', assessment.discrepant_source_count],
    ['Context mismatches', assessment.context_mismatch_count],
    ['Inconclusive sources', assessment.inconclusive_source_count],
    ['No numerical evidence', assessment.no_numerical_evidence_count],
  ] as const

  return (
    <article className="numerical-record-card">
      <div className="numerical-record-header">
        <div className="numerical-record-icon">
          <FlaskConical size={17} />
        </div>

        <div className="numerical-record-heading">
          <span>{claim.claim_id || `CLAIM ${index + 1}`}</span>
          <h3>Numerical verification</h3>
        </div>

        <span
          className={`verdict-badge ${getStatusClass(assessment.verdict)}`}
        >
          {formatVerdict(assessment.verdict)}
        </span>
      </div>

      <p className="numerical-claim-text">{getClaimText(claim)}</p>

      <div className="numerical-reason-panel">
        <StatusIcon size={17} />
        <p>
          {assessment.reason || 'No assessment explanation was returned.'}
        </p>
      </div>

      <div className="numerical-metric-grid">
        {metrics.map(([label, value]) => (
          <div className="numerical-metric" key={label}>
            <span>{label}</span>
            <strong>{value ?? '—'}</strong>
          </div>
        ))}
      </div>

      <div className="numerical-safety-row">
        <div>
          <span>Clinical truth established</span>
          <strong>
            {assessment.clinical_truth_established === undefined
              ? 'Not specified'
              : assessment.clinical_truth_established
                ? 'Yes'
                : 'No'}
          </strong>
        </div>

        <div>
          <span>Medical contradiction established</span>
          <strong>
            {assessment.medical_contradiction_established === undefined
              ? 'Not specified'
              : assessment.medical_contradiction_established
                ? 'Yes'
                : 'No'}
          </strong>
        </div>
      </div>

      {assessment.source_assessments &&
        assessment.source_assessments.length > 0 && (
          <details className="source-assessment-details">
            <summary>Inspect source assessments</summary>
            <pre>
              {JSON.stringify(assessment.source_assessments, null, 2)}
            </pre>
          </details>
        )}
    </article>
  )
}

function NumericalAnalysis() {
  const { response, question } = useAnalysisContext()

  if (!response) {
    return (
      <>
        <PageHeader
          eyebrow="INTELLIGENCE / NUMERICAL VERIFICATION"
          title="Numerical Analysis"
          description="Examine numerical claims, unit normalization, clinical context, and cross-source consistency."
        />

        <div className="content-panel">
          <EmptyState
            icon={FlaskConical}
            title="No numerical analysis available"
            description="Run a research query from the Workspace to inspect its numerical verification results."
          />
        </div>
      </>
    )
  }

  const assessments = response.claims.flatMap((claim, index) => {
    const assessment = claim.numerical_assessment

    if (!assessment) return []

    return [{
      claim,
      index,
      assessment,
    }]
  })

  const countVerdict = (verdict: string) =>
    assessments.filter(
      (item) => item.assessment.verdict === verdict,
    ).length

  const metrics = [
    {
      label: 'Numerical assessments',
      value: assessments.length,
      icon: FlaskConical,
    },
    {
      label: 'Numerically consistent',
      value: countVerdict('NUMERICALLY_CONSISTENT'),
      icon: CheckCircle2,
    },
    {
      label: 'Numerical discrepancies',
      value: countVerdict('NUMERICAL_DISCREPANCY'),
      icon: XCircle,
    },
    {
      label: 'Context mismatches',
      value: countVerdict('CONTEXT_MISMATCH'),
      icon: AlertTriangle,
    },
    {
      label: 'Inconclusive',
      value: countVerdict('INCONCLUSIVE'),
      icon: CircleHelp,
    },
    {
      label: 'No numerical evidence',
      value: countVerdict('NO_NUMERICAL_EVIDENCE'),
      icon: Info,
    },
  ]

  return (
    <>
      <PageHeader
        eyebrow="INTELLIGENCE / NUMERICAL VERIFICATION"
        title="Numerical Analysis"
        description="Examine numerical claims, unit normalization, clinical context, and cross-source consistency."
      />

      <div className="numerical-query-panel">
        <span>ANALYZED QUERY</span>
        <p>{question || 'Previous research query'}</p>
      </div>

      <section className="numerical-overview-grid">
        {metrics.map((metric) => {
          const Icon = metric.icon

          return (
            <div className="metric-card" key={metric.label}>
              <div className="metric-card-top">
                <span className="metric-label">{metric.label}</span>
                <div className="metric-icon">
                  <Icon size={17} />
                </div>
              </div>

              <div className="metric-value">{metric.value}</div>
              <div className="metric-description">
                Count of claim-level assessments
              </div>
            </div>
          )
        })}
      </section>

      <div className="section-heading evidence-section-heading">
        <div>
          <h2>Claim-level numerical assessments</h2>
          <p>
            Each assessment is displayed independently. Counts are not
            treated as independent documents across different claims.
          </p>
        </div>
      </div>

      {assessments.length > 0 ? (
        <div className="numerical-record-list">
          {assessments.map((item) => (
            <AssessmentCard
              key={item.claim.claim_id || item.index}
              claim={item.claim}
              index={item.index}
              assessment={item.assessment}
            />
          ))}
        </div>
      ) : (
        <div className="content-panel">
          <EmptyState
            icon={FlaskConical}
            title="No numerical assessments returned"
            description="The response did not include numerical assessment objects for its claims."
          />
        </div>
      )}

      <div className="evidence-limitation">
        <Info size={17} />
        <div>
          <strong>Interpretation and limitations</strong>
          <p>
            Numerical verification is a consistency analysis, not a
            determination of clinical truth. A missing numerical
            assessment, context mismatch, or inconclusive result must
            not be interpreted as proof that a claim is false.
          </p>
        </div>
      </div>
    </>
  )
}

export default NumericalAnalysis
