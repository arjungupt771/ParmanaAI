
import { Settings as SettingsIcon, Server, ShieldCheck } from 'lucide-react'
import PageHeader from '../components/ui/PageHeader'

function Settings() {
  return (
    <>
      <PageHeader
        eyebrow="CONFIGURATION"
        title="Settings"
        description="Review application configuration and system information."
      />

      <div className="settings-grid">
        <section className="content-panel settings-panel">
          <div className="settings-heading">
            <Server size={19} />
            <div>
              <h3>Backend connection</h3>
              <p>Application API configuration</p>
            </div>
          </div>

          <div className="settings-row">
            <span>API base URL</span>
            <code>VITE_API_BASE_URL</code>
          </div>

          <div className="settings-row">
            <span>Connection status</span>
            <span className="neutral-status">Not checked yet</span>
          </div>
        </section>

        <section className="content-panel settings-panel">
          <div className="settings-heading">
            <ShieldCheck size={19} />
            <div>
              <h3>Safety and limitations</h3>
              <p>Responsible use of research assistance</p>
            </div>
          </div>

          <p className="settings-note">
            ParmanaAI is an experimental evidence-analysis application.
            Its outputs are not a diagnosis, treatment recommendation,
            or substitute for qualified medical judgment. The system
            is not clinically validated.
          </p>
        </section>

        <section className="content-panel settings-panel">
          <div className="settings-heading">
            <SettingsIcon size={19} />
            <div>
              <h3>Application information</h3>
              <p>Current frontend environment</p>
            </div>
          </div>

          <div className="settings-row">
            <span>Application</span>
            <strong>ParmanaAI</strong>
          </div>

          <div className="settings-row">
            <span>Frontend</span>
            <strong>React + TypeScript</strong>
          </div>

          <div className="settings-row">
            <span>Build system</span>
            <strong>Vite</strong>
          </div>
        </section>
      </div>
    </>
  )
}

export default Settings
