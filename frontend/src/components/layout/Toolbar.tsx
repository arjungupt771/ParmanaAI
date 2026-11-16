
import { Bell, Command, Menu, Search } from 'lucide-react'

interface TopbarProps {
  onMenuClick: () => void
}

function Topbar({ onMenuClick }: TopbarProps) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button
          className="icon-button mobile-menu-button"
          onClick={onMenuClick}
          aria-label="Open navigation menu"
        >
          <Menu size={20} />
        </button>

        <div className="breadcrumb">
          <span>ParmanaAI</span>
          <span className="breadcrumb-divider">/</span>
          <span className="breadcrumb-current">Workspace</span>
        </div>
      </div>

      <div className="topbar-actions">
        <button className="search-trigger" aria-label="Search">
          <Search size={15} />
          <span>Search workspace</span>
          <kbd><Command size={11} /> K</kbd>
        </button>

        <div className="topbar-divider" />

        <button
          className="icon-button notification-button"
          aria-label="Notifications"
          title="Notifications"
        >
          <Bell size={18} />
        </button>

        <div className="user-avatar" title="Research workspace">
          P
        </div>
      </div>
    </header>
  )
}

export default Topbar
