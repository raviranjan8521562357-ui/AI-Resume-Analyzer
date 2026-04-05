import { useState } from 'react'

const API = 'http://localhost:8000'

// Level badge colors (dark-theme friendly)
const LEVEL_COLORS = {
  'Production': { bg: 'rgba(34,197,94,0.15)', color: '#4ade80', icon: '🚀' },
  'Advanced':   { bg: 'rgba(139,92,246,0.15)', color: '#a78bfa', icon: '⚡' },
  'Intermediate': { bg: 'rgba(59,130,246,0.15)', color: '#60a5fa', icon: '🔧' },
  'Beginner':   { bg: 'rgba(148,163,184,0.12)', color: '#94a3b8', icon: '📘' },
}

const EXP_TYPE_COLORS = {
  'Full-time':  { bg: 'rgba(34,197,94,0.15)', color: '#4ade80' },
  'Internship': { bg: 'rgba(59,130,246,0.15)', color: '#60a5fa' },
  'Contract':   { bg: 'rgba(245,158,11,0.15)', color: '#fbbf24' },
  'Freelance':  { bg: 'rgba(139,92,246,0.15)', color: '#a78bfa' },
}

export default function CandidateDetail({ candidate, onClose, onToggleShortlist }) {
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)

  if (!candidate) return null

  const a = candidate.analysis || {}
  const info = candidate.candidate_info || {}
  const score = a.ats_score || 0
  const isAccept = a.decision?.toLowerCase() === 'accept'

  const sendChat = async () => {
    if (!chatInput.trim() || chatLoading) return
    const q = chatInput.trim()
    setChatInput('')
    setChatMessages(prev => [...prev, { role: 'user', text: q }])
    setChatLoading(true)

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, candidate_id: candidate.id }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Chat failed')
      setChatMessages(prev => [...prev, { role: 'ai', text: data.answer }])
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'ai', text: `Error: ${err.message}` }])
    } finally {
      setChatLoading(false)
    }
  }

  return (
    <div className="detail-overlay" onClick={onClose}>
      <div className="detail-panel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="detail-header">
          <div>
            <h2>{info.candidate_name || candidate.filename?.replace(/\.[^.]+$/, '')}</h2>
            <div className="filename">📎 {candidate.filename}</div>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              className={`btn-icon ${candidate.shortlisted ? 'shortlisted' : ''}`}
              onClick={() => onToggleShortlist(candidate.id, !candidate.shortlisted)}
              title="Toggle shortlist"
            >
              {candidate.shortlisted ? '★' : '☆'}
            </button>
            <button className="btn-close" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* Scores */}
        <div className="detail-scores">
          <div className="detail-score-card">
            <div className={`detail-score-value ${isAccept ? 'score-cell high' : 'score-cell low'}`}>
              {score}
            </div>
            <div className="detail-score-label">ATS Score</div>
          </div>
          <div className="detail-score-card">
            <div className={`detail-score-value ${isAccept ? 'score-cell high' : 'score-cell low'}`}>
              {a.match_percentage || 0}%
            </div>
            <div className="detail-score-label">Match</div>
          </div>
          <div className="detail-score-card">
            <span className={`badge ${isAccept ? 'accept' : 'reject'}`} style={{ fontSize: '0.85rem' }}>
              {isAccept ? '✓ Accept' : '✕ Reject'}
            </span>
            <div className="detail-score-label" style={{ marginTop: '0.5rem' }}>
              Similarity: {((a.similarity_score || 0) * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Explanation */}
        <div className="detail-section">
          <h3>💡 Explanation</h3>
          <div className="detail-explanation">{a.explanation || 'No analysis available.'}</div>
        </div>

        {/* Info line */}
        {(info.education || info.total_experience_years > 0) && (
          <div className="detail-section">
            <h3>📋 Profile</h3>
            <div className="detail-explanation">
              {info.education && <div><strong>Education:</strong> {info.education}</div>}
              {info.total_experience_years > 0 && (
                <div><strong>Experience:</strong> {info.total_experience_years} year{info.total_experience_years !== 1 ? 's' : ''}</div>
              )}
              {info.experience_level && <div><strong>Level:</strong> {info.experience_level}</div>}
              {info.email && <div><strong>Email:</strong> {info.email}</div>}
            </div>
          </div>
        )}

        {/* Skills */}
        <div className="detail-section">
          <h3>🛠 Skills ({info.skills?.length || 0})</h3>
          <div className="skill-tags">
            {info.skills?.length > 0 ? (
              info.skills.map((s, i) => <span key={i} className="skill-tag neutral">{s}</span>)
            ) : (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>No skills extracted</span>
            )}
          </div>
        </div>

        {/* Matched / Missing Skills */}
        {(a.matched_skills?.length > 0 || a.missing_skills?.length > 0) && (
          <div className="detail-section" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <h3 style={{ color: 'var(--accent-green)' }}>✓ Matched</h3>
              <div className="skill-tags">
                {a.matched_skills?.map((s, i) => <span key={i} className="skill-tag matched">{s}</span>)}
              </div>
            </div>
            <div>
              <h3 style={{ color: 'var(--accent-red)' }}>✕ Missing</h3>
              <div className="skill-tags">
                {a.missing_skills?.map((s, i) => <span key={i} className="skill-tag missing">{s}</span>)}
              </div>
            </div>
          </div>
        )}

        {/* Projects — with Level Badges */}
        {info.projects?.length > 0 && (
          <div className="detail-section">
            <h3>🚀 Projects ({info.projects.length})</h3>
            <div className="project-list">
              {info.projects.map((p, i) => {
                const level = p._level || 'Beginner'
                const levelStyle = LEVEL_COLORS[level] || LEVEL_COLORS['Beginner']
                return (
                  <div key={i} className="project-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem' }}>
                      <h4>{p.name || `Project ${i + 1}`}</h4>
                      <span className="level-badge" style={{
                        background: levelStyle.bg,
                        color: levelStyle.color,
                        padding: '0.15rem 0.55rem',
                        borderRadius: '100px',
                        fontSize: '0.68rem',
                        fontWeight: 600,
                        whiteSpace: 'nowrap',
                        flexShrink: 0,
                      }}>
                        {levelStyle.icon} {level}
                      </span>
                    </div>
                    {p.description && <p>{p.description}</p>}
                    {p.tech?.length > 0 && (
                      <div className="tech-tags">
                        {p.tech.map((t, j) => <span key={j} className="tech-tag">{t}</span>)}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Experience / Work History — with Type Badges */}
        {info.internships?.length > 0 && (
          <div className="detail-section">
            <h3>💼 Experience ({info.internships.length})</h3>
            <div className="internship-list">
              {info.internships.map((intern, i) => {
                const expType = intern.experience_type || 'Full-time'
                const typeStyle = EXP_TYPE_COLORS[expType] || EXP_TYPE_COLORS['Full-time']
                return (
                  <div key={i} className="internship-item" style={{ flexDirection: 'column', alignItems: 'stretch', gap: '0.5rem', padding: '0.85rem 1rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <div className="company">{intern.company || 'Unknown'}</div>
                        <div className="role" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          {intern.role || 'N/A'}
                          <span style={{
                            background: typeStyle.bg,
                            color: typeStyle.color,
                            padding: '0.1rem 0.45rem',
                            borderRadius: '100px',
                            fontSize: '0.62rem',
                            fontWeight: 600,
                            whiteSpace: 'nowrap',
                          }}>
                            {expType}
                          </span>
                        </div>
                      </div>
                      {intern.duration && <div className="duration">{intern.duration}</div>}
                    </div>

                    {/* Work done bullet points */}
                    {intern.work_done?.length > 0 && (
                      <ul style={{ margin: '0.3rem 0 0 0', paddingLeft: '1.2rem', fontSize: '0.8rem', color: 'var(--text-body)', lineHeight: '1.5' }}>
                        {intern.work_done.map((item, j) => (
                          <li key={j}>{item}</li>
                        ))}
                      </ul>
                    )}

                    {/* Technologies used */}
                    {intern.technologies_used?.length > 0 && (
                      <div className="tech-tags" style={{ marginTop: '0.25rem' }}>
                        {intern.technologies_used.map((t, j) => (
                          <span key={j} className="tech-tag">{t}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Feedback Section — Strengths / Weaknesses / Suggestions */}
        {a.explanation && (
          <div className="feedback-section">
            {/* Strengths — derived from matched skills */}
            {a.matched_skills?.length > 0 && (
              <div className="feedback-card strengths">
                <div className="feedback-header">
                  <span className="feedback-icon">💪</span> Strengths
                </div>
                <ul>
                  {a.matched_skills.slice(0, 5).map((s, i) => (
                    <li key={i}>Strong proficiency in <strong>{s}</strong></li>
                  ))}
                  {info.total_experience_years > 0 && (
                    <li>{info.total_experience_years}+ year{info.total_experience_years !== 1 ? 's' : ''} of relevant experience ({info.experience_level})</li>
                  )}
                  {info.projects?.length > 0 && (
                    <li>{info.projects.length} project{info.projects.length !== 1 ? 's' : ''} demonstrating practical skills</li>
                  )}
                </ul>
              </div>
            )}

            {/* Weaknesses — derived from missing skills */}
            {a.missing_skills?.length > 0 && (
              <div className="feedback-card weaknesses">
                <div className="feedback-header">
                  <span className="feedback-icon">⚠️</span> Areas for Improvement
                </div>
                <ul>
                  {a.missing_skills.slice(0, 5).map((s, i) => (
                    <li key={i}>Missing required skill: <strong>{s}</strong></li>
                  ))}
                </ul>
              </div>
            )}

            {/* Suggestions — derived from analysis */}
            <div className="feedback-card suggestions">
              <div className="feedback-header">
                <span className="feedback-icon">💡</span> Suggestions
              </div>
              <ul>
                {a.missing_skills?.length > 0 && (
                  <li>Consider upskilling in {a.missing_skills.slice(0, 3).join(', ')} to improve match</li>
                )}
                {(!info.projects || info.projects.length < 3) && (
                  <li>Add more projects to portfolio to demonstrate hands-on experience</li>
                )}
                {(!info.internships || info.internships.length === 0) && (
                  <li>Gain professional experience through internships or freelance work</li>
                )}
                {a.match_percentage < 70 && (
                  <li>Tailor resume keywords to better match the job description</li>
                )}
                {score < 60 && (
                  <li>Consider restructuring resume with stronger action verbs and quantified achievements</li>
                )}
              </ul>
            </div>
          </div>
        )}

        {/* Chat */}
        <div className="detail-chat">
          <h3 style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.75rem' }}>
            💬 Ask about this candidate
          </h3>

          {chatMessages.length > 0 && (
            <div className="chat-messages">
              {chatMessages.map((m, i) => (
                <div key={i} className={`chat-message ${m.role}`}>{m.text}</div>
              ))}
              {chatLoading && (
                <div className="chat-message ai" style={{ opacity: 0.6 }}>
                  <span className="spinner" style={{ width: 12, height: 12, borderWidth: '1.5px' }} />
                  Thinking...
                </div>
              )}
            </div>
          )}

          <div className="chat-input-row">
            <input
              className="chat-input"
              placeholder="e.g. Does this candidate know React?"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat() } }}
              disabled={chatLoading}
            />
            <button className="btn-send" onClick={sendChat} disabled={!chatInput.trim() || chatLoading}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
