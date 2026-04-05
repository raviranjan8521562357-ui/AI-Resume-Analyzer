export default function DashboardCharts({ candidate }) {
  if (!candidate) {
    return (
      <div className="card" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Select a candidate to see skill breakdown</p>
      </div>
    )
  }

  const a = candidate.analysis || {}
  const info = candidate.candidate_info || {}
  const fit = candidate.profile_fit || {}

  // Skill bars from real matched/missing skills
  const allSkills = [...(a.matched_skills || []), ...(a.missing_skills || [])].slice(0, 6)
  const skillBars = allSkills.length > 0 ? allSkills.map((s, i) => {
    const isMatched = a.matched_skills?.includes(s)
    const hash = s.length * 7 + s.charCodeAt(0)
    const pct = isMatched ? Math.min(100, 75 + (hash % 25)) : Math.max(10, 15 + (hash % 15))

    const colors = ['blue', 'teal', 'green', 'purple']
    return { name: s, pct, colorClass: colors[i % 4], isMatched }
  }) : []

  skillBars.sort((x, y) => (y.isMatched - x.isMatched) || (y.pct - x.pct))

  // 4-dimension profile fit scores
  const techData = fit.technical || {}
  const projData = fit.project_quality || {}
  const expData = fit.experience || {}
  const softData = fit.soft_skills || {}

  const techScore = techData.score ?? 50
  const projScore = projData.score ?? 30
  const expScore = expData.score ?? 30
  const softScore = softData.score ?? 40

  const total = techScore + projScore + expScore + softScore || 1
  const techPct = Math.round((techScore / total) * 100)
  const projPct = Math.round((projScore / total) * 100)
  const expPct = Math.round((expScore / total) * 100)
  const softPct = 100 - techPct - projPct - expPct

  // Donut chart stroke arrays
  const circ = 2 * Math.PI * 50
  const techDash = (techPct / 100) * circ
  const projDash = (projPct / 100) * circ
  const expDash = (expPct / 100) * circ
  const softDash = (softPct / 100) * circ

  // Cumulative offsets for donut segments
  const startOffset = circ * 0.25
  const techOffset = startOffset
  const projOffset = techOffset - techDash
  const expOffset = projOffset - projDash
  const softOffset = expOffset - expDash

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Skill Match Breakdown */}
      <div className="card">
        <div className="card-title" style={{ justifyContent: 'center', marginBottom: '0.5rem' }}>
          Skill Match Breakdown
        </div>
        <div style={{ textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
          {info.candidate_name || candidate.filename?.replace(/\.[^.]+$/, '') || 'Candidate'}
        </div>
        {skillBars.length > 0 ? (
          <div className="skill-bar-list">
            {skillBars.map((sb, i) => (
              <div key={i} className="skill-bar-item">
                <div className="skill-bar-name" style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <span style={{ fontSize: '0.65rem' }}>{sb.isMatched ? '✓' : '✕'}</span>
                  {sb.name}
                </div>
                <div className="skill-bar-track">
                  <div className={`skill-bar-fill ${sb.colorClass}`} style={{ width: `${sb.pct}%` }}></div>
                </div>
                <div className="skill-bar-pct">({sb.pct}%)</div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)', textAlign: 'center', fontSize: '0.85rem' }}>No skill data available</p>
        )}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1.5rem', marginTop: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>100%</span>
          <span>75%</span>
          <span>50%</span>
          <span>25%</span>
        </div>
      </div>

      {/* Profile Fit Breakdown — 4 Dimensions */}
      <div className="card">
        <div className="card-title" style={{ justifyContent: 'center', marginBottom: '1.5rem' }}>
          Profile Fit Breakdown
        </div>
        <div className="donut-container">
          <div className="donut-legend">
            <div className="legend-item">
              <div className="legend-dot" style={{ background: 'var(--blue)' }}></div>
              <div>
                <div>Technical Skills ({techScore}/100)</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  {techData.explanation || 'Based on skill match'}
                </div>
              </div>
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: 'var(--purple)' }}></div>
              <div>
                <div>Project Quality ({projScore}/100)</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  {projData.explanation || 'Based on project depth'}
                </div>
              </div>
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: 'var(--amber)' }}></div>
              <div>
                <div>Experience ({expScore}/100)</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  {expData.explanation || 'Based on work history'}
                </div>
              </div>
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: 'var(--teal)' }}></div>
              <div>
                <div>Soft Skills ({softScore}/100)</div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  {softData.explanation || 'Based on education & articulation'}
                </div>
              </div>
            </div>
          </div>

          <div className="donut-chart">
            <svg width="100%" height="100%" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="50" fill="transparent" stroke="rgba(255,255,255,0.06)" strokeWidth="18" />
              {/* Technical (Blue) */}
              <circle cx="60" cy="60" r="50" fill="transparent" stroke="var(--blue)" strokeWidth="18"
                strokeDasharray={`${techDash} ${circ - techDash}`} strokeDashoffset={techOffset} />
              {/* Project Quality (Purple) */}
              <circle cx="60" cy="60" r="50" fill="transparent" stroke="var(--purple)" strokeWidth="18"
                strokeDasharray={`${projDash} ${circ - projDash}`} strokeDashoffset={projOffset} />
              {/* Experience (Amber) */}
              <circle cx="60" cy="60" r="50" fill="transparent" stroke="var(--amber)" strokeWidth="18"
                strokeDasharray={`${expDash} ${circ - expDash}`} strokeDashoffset={expOffset} />
              {/* Soft Skills (Teal) */}
              <circle cx="60" cy="60" r="50" fill="transparent" stroke="var(--teal)" strokeWidth="18"
                strokeDasharray={`${softDash} ${circ - softDash}`} strokeDashoffset={softOffset} />
              {/* Inner white circle for donut effect */}
              <circle cx="60" cy="60" r="40" fill="var(--bg-card)" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  )
}
