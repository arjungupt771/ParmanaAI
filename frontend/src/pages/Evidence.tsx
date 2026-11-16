
import { useMemo, useState } from 'react'
import {
  BookOpen,
  FileText,
  Info,
  Search,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  Database,
  Link2,
} from 'lucide-react'

import PageHeader from '../components/ui/PageHeader'
import EmptyState from '../components/ui/EmptyState'
import { useAnalysisContext } from '../context/AnalysisContext'

import type {
  ClaimRecord,
  EvidenceAssessment,
  EvidenceRecord,
} from '../types/verification'

function formatVerdict(value?: string): string {
  if (!value) return 'Not assessed'

  return value
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

function getVerdictClass(value?: string): string {
  const verdict = value?.toUpperCase() ?? ''

  if (verdict === 'SUPPORTED') return 'verdict-supported'

  if (verdict === 'CONTRADICTED') return 'verdict-contradicted'

  if (
    verdict.includes('INSUFFICIENT') ||
    verdict.includes('INCONCLUSIVE') ||
    verdict.includes('CONTEXT_MISMATCH') ||
    verdict.includes('CONFLICTING')
  ) {
    return 'verdict-caution'
  }

  return 'verdict-neutral'
}

function formatScore(value?: number | null): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(3)
    : 'Not available'
}

function getAssessmentForEvidence(
  claims: ClaimRecord[],
  evidenceId: string,
): Array<{
  claim: ClaimRecord
  assessment: EvidenceAssessment
}> {
  const results: Array<{
    claim: ClaimRecord
    assessment: EvidenceAssessment
  }> = []

  claims.forEach((claim) => {
    claim.evidence_assessments?.forEach((assessment) => {
      if (assessment.evidence_id === evidenceId) {
        results.push({ claim, assessment })
      }
    })
  })

  return results
}

function EvidenceCard({
  evidence,
  claims,
}: {
  evidence: EvidenceRecord
  claims: ClaimRecord[]
}) {
  const [expanded, setExpanded] = useState(false)

  const assessments = useMemo(
    () => getAssessmentForEvidence(claims, evidence.evidence_id),
    [claims, evidence.evidence_id],
  )

  return (
    <article className="trace-evidence-card">
      <div className="trace-evidence-header">
        <div className="trace-evidence-icon">
          <FileText size={17} />
        </div>

        <div className="trace-evidence-heading">
          <span>{evidence.evidence_id}</span>
          <h3>{evidence.source_type || 'Unknown source type'}</h3>
        </div>

        <span className="evidence-type-badge">
          {evidence.match_type || 'Match unavailable'}
        </span>
      </div>

      <div className="trace-evidence-metadata">
        <div>
          <span>Document ID</span>
          <strong>{evidence.document_id || 'Not available'}</strong>
        </div>

        <div>
          <span>Chunk ID</span>
          <strong>{evidence.chunk_id || 'Not available'}</strong>
        </div>

        <div>
          <span>Retrieval score</span>
          <strong>{formatScore(evidence.retrieval_score)}</strong>
        </div>

        <div>
          <span>Origin</span>
          <strong>{evidence.origin || 'Not available'}</strong>
        </div>

        <div>
          <span>Topic</span>
          <strong>{evidence.topic || 'Not available'}</strong>
        </div>

        <div>
          <span>Verification status</span>
          <strong>{evidence.verification_status || 'Not available'}</strong>
        </div>
      </div>

      <div className="trace-evidence-passage">
        <div className="trace-passage-heading">
          <span>RETRIEVED PASSAGE</span>
          <span>{evidence.text.length} characters</span>
        </div>

        <p className={expanded ? 'passage-expanded' : 'passage-collapsed'}>
          {evidence.text}
        </p>

        <button
          type="button"
          className="trace-expand-button"
          onClick={() => setExpanded((previous) => !previous)}
          aria-expanded={expanded}
        >
          {expanded ? 'Show less' : 'Read full passage'}
          {expanded ? (
            <ChevronUp size={14} />
          ) : (
            <ChevronDown size={14} />
          )}
        </button>
      </div>

      <div className="trace-assessments">
        <div className="trace-assessments-heading">
          <Link2 size={15} />
          <h4>Claim assessment relationships</h4>
          <span>{assessments.length}</span>
        </div>

        {assessments.length > 0 ? (
          <div className="trace-assessment-list">
            {assessments.map(({ claim, assessment }, index) => (
              <div
                className="trace-assessment-item"
                key={`${claim.claim_id || 'claim'}-${index}`}
              >
                <div className="trace-assessment-top">
                  <strong>{claim.claim_id || `Claim ${index + 1}`}</strong>

                  <span
                    className={`verdict-badge ${getVerdictClass(
                      assessment.nli_verdict,
                    )}`}
                  >
                    {formatVerdict(assessment.nli_verdict)}
                  </span>

                  {assessment.is_primary_evidence && (
                    <span className="trace-primary-label">
                      Primary evidence
                    </span>
                  )}
                </div>

                <p>
                  {claim.claim_text ||
                    claim.claim ||
                    'Claim text unavailable'}
                </p>

                {assessment.nli_scores && (
                  <div className="trace-score-row">
                    <span>
                      Entailment: {formatScore(assessment.nli_scores.entailment)}
                    </span>
                    <span>
                      Contradiction: {formatScore(
                        assessment.nli_scores.contradiction,
                      )}
                    </span>
                    <span>
                      Neutral: {formatScore(assessment.nli_scores.neutral)}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="trace-no-assessment">
            No claim-level assessment references this evidence record.
          </p>
        )}
      </div>

      <div className="trace-evidence-footer">
        <ShieldCheck size={14} />
        <span>
          Retrieved evidence is not automatically supporting evidence.
          Assessments above are specific to their associated claims.
        </span>
      </div>

      <details className="trace-provenance-details">
        <summary>Inspect raw provenance metadata</summary>
        <pre>
          {JSON.stringify(evidence.provenance ?? {}, null, 2)}
        </pre>
      </details>
    </article>
  )
}

function Evidence() {
  const { response, question } = useAnalysisContext()

  const evidenceRecords = response?.evidence ?? []
  const claims = response?.claims ?? []

  const legacyClaims = claims.filter(
    (claim) =>
      Boolean(claim.citation) ||
      Boolean(claim.source_type) ||
      Boolean(claim.supporting_source),
  )

  const evidenceSummary = useMemo(() => {
    return {
      total: evidenceRecords.length,
      mapped: evidenceRecords.filter((evidence) =>
        claims.some((claim) =>
          claim.evidence_assessments?.some(
            (assessment) =>
              assessment.evidence_id === evidence.evidence_id,
          ),
        ),
      ).length,
      unmapped: evidenceRecords.filter(
        (evidence) =>
          !claims.some((claim) =>
            claim.evidence_assessments?.some(
              (assessment) =>
                assessment.evidence_id === evidence.evidence_id,
            ),
          ),
      ).length,
    }
  }, [evidenceRecords, claims])

  return (
    <>
      <PageHeader
        eyebrow="INTELLIGENCE / RETRIEVAL"
        title="Evidence Explorer"
        description="Inspect retrieved passages, provenance metadata, and claim-level verification relationships."
      />

      {!response ? (
        <div className="content-panel">
          <EmptyState
            icon={BookOpen}
            title="No analysis available"
            description="Submit a research question in the Workspace first. Its evidence trace will appear here."
          />
        </div>
      ) : (
        <>
          <section className="evidence-query-panel">
            <div className="evidence-query-icon">
              <Search size={17} />
            </div>

            <div>
              <span>ANALYZED QUERY</span>
              <p>{question || 'Previous research query'}</p>
            </div>
          </section>

          <div className="evidence-page-summary">
            <div>
              <span>Retrieved passages</span>
              <strong>{evidenceSummary.total}</strong>
            </div>

            <div>
              <span>Passages mapped to claims</span>
              <strong>{evidenceSummary.mapped}</strong>
            </div>

            <div>
              <span>Unmapped passages</span>
              <strong>{evidenceSummary.unmapped}</strong>
            </div>
          </div>

          {evidenceRecords.length > 0 ? (
            <>
              <div className="section-heading evidence-section-heading">
                <div>
                  <h2>Retrieved evidence passages</h2>
                  <p>
                    Complete passages supplied to the generation and
                    verification pipeline.
                  </p>
                </div>
              </div>

              <div className="trace-evidence-list">
                {evidenceRecords.map((evidence) => (
                  <EvidenceCard
                    key={evidence.evidence_id}
                    evidence={evidence}
                    claims={claims}
                  />
                ))}
              </div>

              <div className="evidence-limitation">
                <Info size={17} />
                <div>
                  <strong>Interpretation and provenance</strong>
                  <p>
                    Retrieval scores describe the retriever's matching
                    output. NLI assessments describe model comparisons
                    between a passage and a specific claim. Neither
                    establishes clinical truth. Source URLs, publication
                    details, and external validation are not provided
                    unless the backend explicitly supplies them.
                  </p>
                </div>
              </div>
            </>
          ) : legacyClaims.length > 0 ? (
            <>
              <div className="section-heading evidence-section-heading">
                <div>
                  <h2>Legacy citation information</h2>
                  <p>
                    This response contains citation excerpts but does
                    not include the new full evidence trace.
                  </p>
                </div>
              </div>

              <div className="evidence-record-list">
                {legacyClaims.map((claim, index) => (
                  <article
                    className="evidence-record-card"
                    key={claim.claim_id || index}
                  >
                    <div className="evidence-record-header">
                      <div className="evidence-record-icon">
                        <FileText size={17} />
                      </div>

                      <div className="evidence-record-title">
                        <span>{claim.claim_id || `CLAIM ${index + 1}`}</span>
                        <h3>
                          {claim.source_type || 'Source type unavailable'}
                        </h3>
                      </div>

                      <span className="evidence-type-badge">
                        {claim.citation_match || 'Unspecified match'}
                      </span>
                    </div>

                    <div className="evidence-claim">
                      <span>ASSOCIATED CLAIM</span>
                      <p>
                        {claim.claim_text ||
                          claim.claim ||
                          'Claim text unavailable'}
                      </p>
                    </div>

                    {claim.citation && (
                      <blockquote className="evidence-quote">
                        <span>CITATION EXCERPT</span>
                        {claim.citation}
                      </blockquote>
                    )}
                  </article>
                ))}
              </div>

              <div className="evidence-limitation">
                <Info size={17} />
                <div>
                  <strong>Full evidence trace unavailable</strong>
                  <p>
                    This response uses the older API format. Full
                    retrieved passages and claim-to-evidence IDs will
                    appear after the backend evidence-trace changes
                    are applied.
                  </p>
                </div>
              </div>
            </>
          ) : (
            <div className="content-panel">
              <EmptyState
                icon={Database}
                title="No evidence records returned"
                description="The current response contains neither full evidence records nor legacy citation information."
              />
            </div>
          )}
        </>
      )}
    </>
  )
}

export default Evidence
