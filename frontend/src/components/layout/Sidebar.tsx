
import {
  BarChart3,
  BookOpen,
  BrainCircuit,
  FileCheck2,
  FlaskConical,
  LayoutDashboard,
  Settings,
  ShieldCheck,
} from 'lucide-react'
import { NavLink } from 'react-router-dom'

interface SidebarProps {
  mobileMenuOpen: boolean
  onNavigate: () => void
}

const navigation = [
  {
    label: 'WORKSPACE',
    items: [
      { name: 'Overview', path: '/', icon: LayoutDashboard },
    ],
  },
  {
    label: 'INTELLIGENCE',
    items: [
      { name: 'Claims', path: '/claims', icon: FileCheck2 },
      { name: 'Evidence', path: '/evidence', icon: BookOpen },
      { name: 'Numerical Analysis', path: '/numerical', icon: FlaskConical },
    ],
  },
  {
    label: 'QUALITY',
    items: [
      { name: 'Evaluation', path: '/evaluation', icon: BarChart3 },
    ],
  },
]

function Sidebar({ mobileMenuOpen, onNavigate }: SidebarProps) {
  return (
    <aside className={`sidebar ${mobileMenuOpen ? 'sidebar-open' : ''}`}>
      <div className="sidebar-brand">
        <div className="brand-icon">
          <BrainCircuit size={22} />
        </div>

        <div className="brand-copy">
          <span className="brand-name">ParmanaAI</span>
          <span className="brand-caption">EVIDENCE INTELLIGENCE</span>
        </div>
      </div>

      <div className="sidebar-workspace">
        <div className="workspace-avatar">P</div>
        <div className="workspace-copy">
          <span className="workspace-name">Research Workspace</span>
          <span className="workspace-plan">Personal environment</span>
        </div>
      </div>

      <nav className="sidebar-navigation">
        {navigation.map((group) => (
          <div className="nav-group" key={group.label}>
            <span className="nav-group-label">{group.label}</span>

            {group.items.map((item) => {
              const Icon = item.icon

              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    `nav-link ${isActive ? 'nav-link-active' : ''}`
                  }
                >
                  <Icon size={17} strokeWidth={1.8} />
                  <span>{item.name}</span>
                </NavLink>
              )
            })}
          </div>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <div className="sidebar-safety">
          <div className="safety-icon">
            <ShieldCheck size={17} />
          </div>
          <div>
            <span className="safety-title">Research assistance</span>
            <span className="safety-caption">Not a clinical decision tool</span>
          </div>
        </div>

        <NavLink
          to="/settings"
          onClick={onNavigate}
          className={({ isActive }) =>
            `nav-link settings-link ${isActive ? 'nav-link-active' : ''}`
          }
        >
          <Settings size={17} strokeWidth={1.8} />
          <span>Settings</span>
        </NavLink>

        <div className="sidebar-version">
          <span className="version-dot" />
          ParmanaAI · Development
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
