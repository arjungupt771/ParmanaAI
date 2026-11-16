
import { useState } from 'react'
import { Outlet } from 'react-router-dom'

import Sidebar from './Sidebar'
import Topbar from './Topbar'

function AppShell() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className="app-shell">
      {mobileMenuOpen && (
        <button
          className="mobile-overlay"
          aria-label="Close navigation menu"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      <Sidebar
        mobileMenuOpen={mobileMenuOpen}
        onNavigate={() => setMobileMenuOpen(false)}
      />

      <div className="app-main">
        <Topbar onMenuClick={() => setMobileMenuOpen(true)} />

        <main className="page-container">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default AppShell
