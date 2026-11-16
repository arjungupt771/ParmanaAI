
import type { LucideIcon } from 'lucide-react'

interface MetricCardProps {
  label: string
  value: string
  description: string
  icon: LucideIcon
}

function MetricCard({
  label,
  value,
  description,
  icon: Icon,
}: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        <div className="metric-icon">
          <Icon size={18} />
        </div>
      </div>

      <div className="metric-value">{value}</div>
      <div className="metric-description">{description}</div>
    </div>
  )
}

export default MetricCard
