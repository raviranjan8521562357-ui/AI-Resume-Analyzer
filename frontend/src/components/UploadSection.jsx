import { useRef, useState } from 'react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export default function UploadSection({ onAnalysisComplete, setError }) {
  const [files, setFiles] = useState([])
  const [jd, setJd] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const inputRef = useRef(null)

  const handleFiles = (newFiles) => {
    const valid = Array.from(newFiles).filter(f => {
      const ext = f.name.toLowerCase()
      return ext.endsWith('.pdf') || ext.endsWith('.docx')
    })
    if (valid.length < newFiles.length) setError('Some files skipped — only PDF/DOCX allowed.')
    // Deduplicate by filename — prevent adding the same file twice
    setFiles(prev => {
      const existingNames = new Set(prev.map(f => f.name))
      const unique = valid.filter(f => !existingNames.has(f.name))
      if (unique.length < valid.length) {
        setError('Duplicate files skipped — each file can only be added once.')
      }
      return [...prev, ...unique]
    })
  }

  const removeFile = (i) => setFiles(prev => prev.filter((_, idx) => idx !== i))

  const isSubmitting = useRef(false)

  const handleAnalyze = async () => {
    if (!files.length || !jd.trim()) {
      setError('Upload resumes and enter a job description.')
      return
    }
    // Prevent double-submission
    if (isSubmitting.current) return
    isSubmitting.current = true
    
    setLoading(true)
    setError(null)

    try {
      // Clear previous candidates to prevent duplicates from accumulation
      setProgress('Preparing...')
      try {
        await fetch(`${API}/candidates`, { method: 'DELETE' })
      } catch (clearErr) {
        // If clearing fails, backend may be down — continue and let the upload catch it
        console.warn('Could not clear candidates:', clearErr.message)
      }

      setProgress('Uploading resumes...')
      const formData = new FormData()
      files.forEach(f => formData.append('files', f))
      let upRes
      try {
        upRes = await fetch(`${API}/upload-multiple`, { method: 'POST', body: formData })
      } catch (networkErr) {
        throw new Error(`Cannot connect to backend at ${API}. Please check that the backend is running and CORS is configured.`)
      }
      if (!upRes.ok) {
        const errBody = await upRes.json().catch(() => ({}))
        throw new Error(errBody.detail || `Upload failed (HTTP ${upRes.status})`)
      }
      const upData = await upRes.json()
      if (upData.uploaded === 0) throw new Error('No resumes could be processed.')

      setProgress(`Analyzing ${upData.uploaded} resume${upData.uploaded > 1 ? 's' : ''} with AI...`)
      let aRes
      try {
        aRes = await fetch(`${API}/analyze-batch`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_description: jd }),
        })
      } catch (networkErr) {
        throw new Error(`Connection lost during analysis. Please check your backend at ${API}.`)
      }
      if (!aRes.ok) {
        const errBody = await aRes.json().catch(() => ({}))
        throw new Error(errBody.detail || `Analysis failed (HTTP ${aRes.status})`)
      }
      const results = await aRes.json()
      onAnalysisComplete(results.candidates)
      setFiles([])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setProgress('')
      isSubmitting.current = false
    }
  }

  if (loading) {
    return (
      <div className="loading-state">
        <div className="spinner-lg" />
        <p>{progress}</p>
        <p className="loading-sub">This may take a moment for multiple resumes</p>
      </div>
    )
  }

  return (
    <>
      <div className="upload-row">
        {/* Job Description */}
        <div className="card">
          <div className="card-title">📋 Upload Job Description</div>
          <div 
            className="upload-box" 
            style={{ padding: '1rem', textAlign: 'left', cursor: 'default', borderStyle: 'solid', borderColor: 'var(--border)', background: 'var(--bg-light)' }}
          >
            <textarea
              className="jd-input"
              placeholder="Paste the job description here...&#10;&#10;Example: We are looking for a Full Stack Developer with 2+ years experience in React, Node.js, Python, AWS..."
              value={jd}
              onChange={(e) => setJd(e.target.value)}
              style={{ minHeight: '230px', border: 'none', background: 'transparent', padding: 0 }}
            />
          </div>
        </div>

        {/* Resume Upload */}
        <div className="card">
          <div className="card-title">📄 Upload Resumes (Multiple PDF/DOCX)</div>
          <div
            className={`upload-box ${files.length ? 'active' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); handleFiles(e.dataTransfer.files) }}
          >
            <div className="upload-icon">{files.length ? '✅' : '📁'}</div>
            <div className="upload-label">
              {files.length ? `${files.length} file${files.length > 1 ? 's' : ''} selected` : 'Drop files here'}
            </div>
            <div className="upload-sub">PDF or DOCX • Multiple files supported</div>
            <button className={`btn-browse ${files.length ? 'green' : ''}`} onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}>
              Browse File
            </button>
            <input
              ref={inputRef}
              type="file" accept=".pdf,.docx" multiple
              style={{ display: 'none' }}
              onChange={(e) => { handleFiles(e.target.files); e.target.value = '' }}
            />
          </div>
          {files.length > 0 && (
            <div className="file-chips">
              {files.map((f, i) => (
                <span key={i} className="file-chip">
                  📎 {f.name}
                  <button onClick={() => removeFile(i)}>✕</button>
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      <button className="btn-analyze" onClick={handleAnalyze} disabled={!files.length || !jd.trim()}>
        🔍 Analyze Resumes
      </button>
    </>
  )
}
