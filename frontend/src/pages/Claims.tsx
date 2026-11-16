import { useMemo, useState } from 'react'
import { FileCheck2 } from 'lucide-react'

import PageHeader from '../components/ui/PageHeader'
import EmptyState from '../components/ui/EmptyState'
import ClaimResultCard from '../components/claims/ClaimResultCard'
import { useAnalysisContext } from '../context/AnalysisContext'

const filters = [
  'ALL',
  'SUPPORTED',
  'CONTRADICTED',
  'INSUFFICIENT_EVIDENCE',
  'INCONCLUSIVE',
  'OTHER',
] as const

type Filter = (typeof filters)[number]

function Claims() {
  const { response, question } = useAnalysisContext()
  const [activeFilter, setActiveFilter] = useState<Filter>('ALL')

  const claims = response?.claims ?? []

  const filteredClaims = useMemo(() => {
    return claims.filter((claim) => {
      const verdict = (
        claim.combined_verdict ??
        claim.verdict ??
        'OTHER'
      ).toUpperCase()

      if (activeFilter === 'ALL') return true
      if (activeFilter === 'OTHER') {
        return ![
          'SUPPORTED',
          'CONTRADICTED',
          'INSUFFICIENT_EVIDENCE',
          'INCONCLUSIVE',
        ].includes(verdict)
      }

      if (activeFilter === 'INCONCLUSIVE') {
        return verdict.includes('INCONCLUSIVE') ||
          verdict.includes('CONTEXT_MISMATCH') ||
          verdict.includes('CONFLICTING')
      }

      return verdict === activeFilter
    })
  }, [claims, activeFilter])

  return (
    <>
      <PageHeader
        eyebrow="INTELLIGENCE / VERIFICATION"
        title="Claim Explorer"
        description="Inspect extracted claims and their individual NLI and numerical verification signals."
      />

      {!response ? (
        <div className="content-panel">
          <EmptyState
            icon={FileCheck2}
            title="No claims available"
            description="Run a research query in Workspace to inspect the returned claim-level verification results."
          />
        </div>
      ) : (
        <>
          <div className="numerical-query-panel">
            <span>ANALYZED QUERY</span>
            <p>{question || 'Previous research query'}</p>
          </div>

          <div className="claims-summary">
            <div>
              <span>Total claims</span>
              <strong>{claims.length}</strong>
            </div>

            <div>
              <span>Filtered claims</span>
              <strong>{filteredClaims.length}</strong>
            </div>

            <div>
              <span>Hallucination heuristic</span>
              <strong>{response.hallucination_score.toFixed(3)}</strong>
            </div>
          </div>

          <div className="filter-bar">
            <span className="filter-label">VERDICT</span>

            {filters.map((filter) => (
              <button
                key={filter}
                className={`filter-chip ${activeFilter === filter ? 'filter-chip-active' : ''}`}
                onClick={() => setActiveFilter(filter)}
              >
                {filter.replaceAll('_', ' ')}
              </button>
            ))}
          </div>

          {filteredClaims.length > 0 ? (
            <div className="claim-results-list">
              {filteredClaims.map((claim, index) => (
                <ClaimResultCard
                  key={claim.claim_id || index}
                  claim={claim}
                  index={index}
                />
              ))}
            </div>
          ) : (
            <div className="content-panel">
              <EmptyState
                icon={FileCheck2}
                title="No claims match this filter"
                description="Try another verdict category or select all claims."
              />
            </div>
          )}
        </>
      )}
    </>
  )
}

export default Claims
