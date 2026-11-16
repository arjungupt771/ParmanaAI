
import { useState } from 'react'
import {
  ChevronDown,
  ChevronUp,
  FileCheck2,
  BookOpen,
  FlaskConical,
  Activity,
  Info,
} from 'lucide-react'

import type {
  ClaimRecord,
  NliScores,
  NumericalAssessment,
} from '../../types/verification'

interface ClaimResultCardProps {
  claim: ClaimRecord
  index: number
}

function formatVerdict(value?: string): string {
  if (!value) return 'Not available'

  return value
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

function getVerdictClass(value?: string): string {
  const verdict = value?.toUpperCase() ?? ''

  if (verdict === 'SUPPORTED' || verdict === 'NUMERICALLY_CONSISTENT') {
    return 'verdict-supported'
  }

  if (verdict === 'CONTRADICTED' || verdict === 'NUMERICAL_DISCREPANCY') {
    return 'verdict-contradicted'
  }

  if (
    verdict.includes('INSUFFICIENT') ||
    verdict.includes('INCONCLUSIVE') ||
    verdict.includes('CONTEXT_MISMATCH') ||
    verdict.includes('CONFLICTING') ||
    verdict.includes('NO_NUMERICAL')
  ) {
    return 'verdict-caution'
  }

  return 'verdict-neutral'
}

function getClaimText(claim: ClaimRecord): string {
  return (
    claim.claim_text ||
    claim.claim ||
    claim.original_sentence ||
    'Claim text unavailable'
  )
}

function ScoreBar({
  label,
  value,
}: {
  label: string
  value?: number
}) {
  const safeValue =
    typeof value === 'number' && Number.isFinite(value)
      ? Math.min(1, Math.max(0, value))
      : null

  return (
    <div className="score-row">
      <div className="score-label">
        <span>{label}</span>
        <strong>
          {safeValue === null ? 'N/A' : safeValue.toFixed(4)}
        </strong>
      </div>

      <div className="score-track">
        <div
          className="score-fill"
          style={{
            width: `${(safeValue ?? 0) * 100}%`,
          }}
        />
      </div>
    </div>
  )
}

function NumericalDetails({
  assessment,
}: {
  assessment?: NumericalAssessment
}) {
  if (!assessment) {
    return (
      <p className="detail-unavailable">
        No numerical assessment was returned.
      </p>
    )
  }

  const metrics = [
    ['Source assessments', assessment.source_assessment_count],
    ['Unique documents', assessment.unique_document_count],
    ['Consistent sources', assessment.consistent_source_count],
    ['Discrepant sources', assessment.discrepant_source_count],
    ['Context mismatches', assessment.context_mismatch_count],
    ['Inconclusive sources', assessment.inconclusive_source_count],
  ] as const

  return (
    <div className="numerical-details">
      <div className="detail-verdict-row">
        <span>Numerical verdict</span>
        <span className={`verdict-badge ${getVerdictClass(assessment.verdict)}`}>
          {formatVerdict(assessment.verdict)}
        </span>
      </div>

      {assessment.reason && (
        <p className="numerical-reason">{assessment.reason}</p>
      )}

      <div className="numerical-count-grid">
        {metrics.map(([label, value]) => (
          <div className="numerical-count" key={label}>
            <span>{label}</span>
            <strong>{value ?? '—'}</strong>
          </div>
        ))}
      </div>

      <div className="safety-disclaimer">
        <Info size={14} />
        <span>
          This assessment does not establish clinical truth or a medical contradiction.
        </span>
      </div>
    </div>
  )
}


function EvidenceAssessmentDetails({
  claim,
}: {
  claim: ClaimRecord
}) {
  const assessments = claim.evidence_assessments ?? []

  if (assessments.length === 0) {
    return (
      <p className="detail-unavailable">
        No detailed evidence assessments were returned.
      </p>
    )
  }

  return (
    <div className="claim-evidence-assessments">
      {assessments.map((assessment, index) => (
        <div
          className="claim-evidence-assessment"
          key={`${assessment.evidence_id || 'evidence'}-${index}`}
        >
          <div className="claim-evidence-assessment-header">
            <strong>
              {assessment.evidence_id || `Evidence ${index + 1}`}
            </strong>

            <span
              className={`verdict-badge ${getVerdictClass(
                assessment.nli_verdict,
              )}`}
            >
              {formatVerdict(assessment.nli_verdict)}
            </span>
          </div>

          {assessment.is_primary_evidence && (
            <span className="trace-primary-label">
              Primary evidence
            </span>
          )}

          <div className="claim-evidence-metadata">
            <span>
              Document: {assessment.document_id || 'Unavailable'}
            </span>
            <span>
              Chunk: {assessment.chunk_id || 'Unavailable'}
            </span>
            <span>
              Contradiction relevance guard:{' '}
              {assessment.contradiction_guard_passed === undefined
                ? 'Not available'
                : assessment.contradiction_guard_passed
                  ? 'Passed'
                  : 'Not passed'}
            </span>
          </div>

          {assessment.nli_scores && (
            <div className="claim-evidence-scores">
              <ScoreBar
                label="Entailment"
                value={assessment.nli_scores.entailment}
              />
              <ScoreBar
                label="Contradiction"
                value={assessment.nli_scores.contradiction}
              />
              <ScoreBar
                label="Neutral"
                value={assessment.nli_scores.neutral}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}


function ClaimResultCard({
  claim,
  index,
}: ClaimResultCardProps) {
  const [expanded, setExpanded] = useState(false)

  const signals = claim.verification_signals
  const nliScores: NliScores | undefined =
    signals?.nli?.scores ?? claim.nli_scores

  const numericalAssessment = claim.numerical_assessment

  const combinedVerdict =
    claim.combined_verdict ?? claim.verdict

  const nliVerdict =
    claim.nli_verdict ?? signals?.nli?.verdict ?? claim.verdict

  return (
    <article className="claim-result-card">
      <div className="claim-result-heading">
        <div className="claim-number">
          <FileCheck2 size={15} />
          <span>{claim.claim_id || `CLAIM ${index + 1}`}</span>
        </div>

        <span className={`verdict-badge ${getVerdictClass(combinedVerdict)}`}>
          {formatVerdict(combinedVerdict)}
        </span>
      </div>

      <p className="claim-result-text">
        {getClaimText(claim)}
      </p>

      <div className="claim-signal-grid">
        <div className="claim-signal">
          <span>NLI verdict</span>
          <strong>{formatVerdict(nliVerdict)}</strong>
        </div>

        <div className="claim-signal">
          <span>Numerical verdict</span>
          <strong>
            {formatVerdict(numericalAssessment?.verdict)}
          </strong>
        </div>

        <div className="claim-signal">
          <span>Extraction type</span>
          <strong>
            {formatVerdict(claim.extraction_type)}
          </strong>
        </div>
      </div>

      {nliScores && (
        <section className="claim-detail-section">
          <div className="detail-section-title">
            <Activity size={15} />
            <h4>NLI model scores</h4>
          </div>

          <div className="nli-score-panel">
            <ScoreBar
              label="Entailment"
              value={nliScores.entailment}
            />
            <ScoreBar
              label="Contradiction"
              value={nliScores.contradiction}
            />
            <ScoreBar
              label="Neutral"
              value={nliScores.neutral}
            />
          </div>

          <p className="detail-footnote">
            Model output scores, not calibrated medical confidence.
          </p>
        </section>
      )}

      <section className="claim-detail-section">
        <div className="detail-section-title">
          <FlaskConical size={15} />
          <h4>Numerical assessment</h4>
        </div>

        <NumericalDetails assessment={numericalAssessment} />
      </section>

      <section className="claim-detail-section">
        <div className="detail-section-title">
          <BookOpen size={15} />
          <h4>Evidence information</h4>
        </div>

        <div className="evidence-summary">
          <div className="evidence-meta-grid">
            <div>
              <span>Source type</span>
              <strong>{claim.source_type || 'Not available'}</strong>
            </div>

            <div>
              <span>Supporting source label</span>
              <strong>{claim.supporting_source || 'Not available'}</strong>
            </div>

            <div>
              <span>Citation match</span>
              <strong>{claim.citation_match || 'Not available'}</strong>
            </div>

            <div>
              <span>Source count</span>
              <strong>
                {claim.supporting_sources ?? '—'} / {claim.total_sources ?? '—'}
              </strong>
            </div>
          </div>

          {claim.citation && (
            <blockquote className="citation-excerpt">
              <span>Returned citation excerpt</span>
              {claim.citation}
            </blockquote>
          )}

          <p className="detail-footnote">
            This is citation information returned by the backend.
            Full document provenance and retrieved passages are not
            included in this response.
          </p>
        </div>
      </section>
      
      <section className="claim-detail-section">
        <div className="detail-section-title">
          <BookOpen size={15} />
          <h4>Claim-to-evidence traceability</h4>
        </div>

        <div className="evidence-summary">
          <div className="claim-primary-evidence">
            <span>Primary evidence ID</span>
            <strong>
              {claim.primary_evidence_id || 'Not available'}
            </strong>
          </div>

          <EvidenceAssessmentDetails claim={claim} />
        </div>
      </section>


      {claim.explanation && (
        <section className="claim-explanation">
          <span>Verification explanation</span>
          <p>{claim.explanation}</p>
        </section>
      )}

      <button
        className="claim-expand-button"
        onClick={() => setExpanded((previous) => !previous)}
        aria-expanded={expanded}
      >
        {expanded ? 'Hide raw verification data' : 'Inspect raw verification data'}

        {expanded ? (
          <ChevronUp size={15} />
        ) : (
          <ChevronDown size={15} />
        )}
      </button>

      {expanded && (
        <div className="claim-raw-details">
          <pre>{JSON.stringify(claim, null, 2)}</pre>
        </div>
      )}
    </article>
  )
}

export default ClaimResultCard
