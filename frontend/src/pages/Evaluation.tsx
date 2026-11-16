
import { BarChart3 } from 'lucide-react'
import PageHeader from '../components/ui/PageHeader'
import EmptyState from '../components/ui/EmptyState'

function Evaluation() {
  return (
    <>
      <PageHeader
        eyebrow="QUALITY / BENCHMARKING"
        title="Evaluation Center"
        description="Review verification quality, benchmark results, and system-level evaluation runs."
      />

      <div className="metrics-grid">
        {[
          ['Evaluation runs', '—'],
          ['Claims evaluated', '—'],
          ['Supported predictions', '—'],
          ['Inconclusive predictions', '—'],
        ].map(([label, value]) => (
          <div className="metric-card" key={label}>
            <span className="metric-label">{label}</span>
            <div className="metric-value">{value}</div>
            <span className="metric-description">No evaluation data yet</span>
          </div>
        ))}
      </div>

      <div className="content-panel">
        <EmptyState
          icon={BarChart3}
          title="No evaluation runs yet"
          description="Benchmark metrics, confusion matrices, and evaluation history will appear here when evaluation data is connected."
        />
      </div>
    </>
  )
}

export default Evaluation
