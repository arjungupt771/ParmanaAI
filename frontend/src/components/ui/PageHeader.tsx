
import type { ReactNode } from 'react'

interface PageHeaderProps {
  eyebrow: string
  title: string
  description: string
  action?: ReactNode
}

function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: PageHeaderProps) {
  return (
    <div className="page-header">
      <div>
        <div className="page-eyebrow">{eyebrow}</div>
        <h1 className="page-title">{title}</h1>
        <p className="page-description">{description}</p>
      </div>

      {action && <div className="page-header-action">{action}</div>}
    </div>
  )
}

export default PageHeader
