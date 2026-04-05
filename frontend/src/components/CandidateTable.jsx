const COLORS = ['#3b82f6', '#06b6d4', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#ef4444', '#f97316', '#a855f7']

function getInitials(name) {
  if (!name || name === 'Unknown') return '?'
  return name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
}

function getStatus(score) {
  if (score >= 70) return { label: 'Match', cls: 'match' }
  if (score >= 50) return { label: 'Moderate', cls: 'moderate' }
  return { label: 'Low', cls: 'low' }
}

function scoreClass(score) {
  if (score >= 70) return 'high'
  if (score >= 50) return 'medium'
  return 'low'
}

export default function CandidateTable({ candidates, onSelectForChart, onViewDetail, onToggleShortlist, filter, activeCandidateId }) {
  const filtered = filter === 'all' ? candidates
    : filter === 'shortlisted' ? candidates.filter(c => c.shortlisted)
    : filter === 'accepted' ? candidates.filter(c => c.analysis?.decision?.toLowerCase() === 'accept')
    : candidates.filter(c => c.analysis?.decision?.toLowerCase() === 'reject')

  if (!filtered.length) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📋</div>
        <h3>No candidates found</h3>
        <p>{filter === 'shortlisted' ? 'Star candidates to shortlist them' : 'Upload resumes to see rankings'}</p>
      </div>
    )
  }

  return (
    <table className="ranking-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Candidate Name</th>
          <th>Similarity Score (NLP)</th>
          <th>ATS Score</th>
          <th>Status</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {filtered.map((c, i) => {
          const a = c.analysis || {}
          const info = c.candidate_info || {}
          const score = a.ats_score || 0
          const similarity = ((a.similarity_score || 0) * 100).toFixed(0)
          const status = getStatus(score)
          const name = info.candidate_name || c.filename?.replace(/\.[^.]+$/, '') || 'Unknown'
          const isActive = c.id === activeCandidateId

          return (
            <tr
              key={c.id}
              onClick={() => onSelectForChart(c)}
              className={isActive ? 'active-row' : ''}
            >
              <td><span className="rank-num">{i + 1}</span></td>
              <td>
                <div className="name-cell">
                  <div className="avatar" style={{ background: COLORS[i % COLORS.length] }}>
                    {getInitials(name)}
                  </div>
                  <div>
                    <div className="name-text">{name}</div>
                    <div className="file-text">{c.filename}</div>
                  </div>
                </div>
              </td>
              <td><span className={`score-display ${scoreClass(parseInt(similarity))}`}>{similarity}%</span></td>
              <td><span className={`score-display ${scoreClass(score)}`}>{score}</span></td>
              <td><span className={`status-badge ${status.cls}`}>{status.label}</span></td>
              <td>
                <div className="action-btns" onClick={(e) => e.stopPropagation()}>
                  <button className="btn-sm" title="View Details" onClick={() => onViewDetail(c)}>👁</button>
                  <button className={`btn-sm ${c.shortlisted ? 'starred' : ''}`} title="Shortlist" onClick={() => onToggleShortlist(c.id, !c.shortlisted)}>
                    {c.shortlisted ? '★' : '☆'}
                  </button>
                </div>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
