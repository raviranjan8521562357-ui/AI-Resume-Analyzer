import { useState } from 'react'
import UploadSection from './components/UploadSection'
import CandidateTable from './components/CandidateTable'
import CandidateDetail from './components/CandidateDetail'
import DashboardCharts from './components/DashboardCharts'

const API = 'http://localhost:8000'

export default function App() {
  const [candidates, setCandidates] = useState([])
  const [chartCandidate, setChartCandidate] = useState(null)    // For skill breakdown charts
  const [detailCandidate, setDetailCandidate] = useState(null)  // For slide-in detail panel
  const [error, setError] = useState(null)

  // Charts show: explicitly selected candidate → or first candidate as default
  const activeCandidate = chartCandidate || (candidates.length > 0 ? candidates[0] : null)

  const handleAnalysisComplete = (results) => {
    setCandidates(results)
    setChartCandidate(null)  // Reset to first candidate
    setDetailCandidate(null)
    setError(null)
  }

  const toggleShortlist = async (id, shortlisted) => {
    try {
      await fetch(`${API}/shortlist`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_id: id, shortlisted }),
      })
      setCandidates(prev => prev.map(c => c.id === id ? { ...c, shortlisted } : c))
      if (detailCandidate?.id === id) {
        setDetailCandidate(prev => ({ ...prev, shortlisted }))
      }
      if (chartCandidate?.id === id) {
        setChartCandidate(prev => ({ ...prev, shortlisted }))
      }
    } catch {
      setError('Failed to update shortlist')
    }
  }

  return (
    <div className="app">
      <div className="app-wrapper">
        {/* Header */}
        <header className="header">
          <h1>Smart Resume Analyzer</h1>
          <p>AI-Powered Applicant Tracking System &amp; Candidate Intelligence</p>
        </header>

        {/* Error Banner */}
        {error && (
          <div className="error-banner">
            ⚠️ {error}
            <button
              onClick={() => setError(null)}
              style={{ float: 'right', background: 'none', border: 'none', color: 'var(--red)', cursor: 'pointer', fontSize: '1rem' }}
            >✕</button>
          </div>
        )}

        {/* Main Grid Layout (Left: Upload + Table, Right: Charts) */}
        <div className="main-grid" style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '2rem', alignItems: 'start' }}>
          
          {/* Left Column: Uploads & Ranking */}
          <div className="left-column" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            
            {/* Upload Section */}
            <div className="upload-section-wrapper">
              <UploadSection onAnalysisComplete={handleAnalysisComplete} setError={setError} />
            </div>

            {/* Ranking Table */}
            <div className="card" style={{ padding: '0', overflow: 'hidden' }}>
              <div className="card-title" style={{ padding: '1.5rem 1.5rem 1rem 1.5rem', marginBottom: 0 }}>
                🏆 Candidate Ranking
              </div>
              <div style={{ padding: '0 1.5rem 1.5rem 1.5rem' }}>
                 <CandidateTable
                    candidates={candidates}
                    filter="all"
                    activeCandidateId={activeCandidate?.id}
                    onSelectForChart={setChartCandidate}
                    onViewDetail={setDetailCandidate}
                    onToggleShortlist={toggleShortlist}
                  />
              </div>
            </div>
          </div>

          {/* Right Column: Charts */}
          <div className="right-column">
            <DashboardCharts candidate={activeCandidate} />
          </div>

        </div>
      </div>

      {/* Slide-in Detail Panel */}
      {detailCandidate && (
        <CandidateDetail
          candidate={detailCandidate}
          onClose={() => setDetailCandidate(null)}
          onToggleShortlist={toggleShortlist}
        />
      )}
    </div>
  )
}
