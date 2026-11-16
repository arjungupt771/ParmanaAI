
import { Navigate, Route, Routes } from 'react-router-dom'

import AppShell from './components/layout/AppShell'
import Workspace from './pages/Workspace'
import Claims from './pages/Claims'
import Evidence from './pages/Evidence'
import NumericalAnalysis from './pages/NumericalAnalysis'
import Evaluation from './pages/Evaluation'
import Settings from './pages/Settings'

function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Workspace />} />
        <Route path="claims" element={<Claims />} />
        <Route path="evidence" element={<Evidence />} />
        <Route path="numerical" element={<NumericalAnalysis />} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default App
