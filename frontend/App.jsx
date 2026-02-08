import { useState, useEffect, useRef } from 'react'

// API base URL
const API_URL = '/api'

// Simple API client
const api = {
  async post(endpoint, data) {
    const res = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    if (!res.ok) throw new Error(`API error: ${res.status}`)
    return res.json()
  },
  
  async get(endpoint) {
    const res = await fetch(`${API_URL}${endpoint}`)
    if (!res.ok) throw new Error(`API error: ${res.status}`)
    return res.json()
  }
}

// Main App
export default function App() {
  const [activeTab, setActiveTab] = useState('assess')
  const [health, setHealth] = useState(null)

  useEffect(() => {
    api.get('/health').then(setHealth).catch(() => setHealth({ status: 'unhealthy' }))
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>NG12 Cancer Risk Assessor</h1>
        <p>Clinical decision support based on NICE NG12 guidelines</p>
      </header>

      {health && (
        <div className="status">
          <span className={`status-dot ${health.status}`}></span>
          <span>API: {health.status}</span>
        </div>
      )}

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'assess' ? 'active' : ''}`}
          onClick={() => setActiveTab('assess')}
        >
          Assessment
        </button>
        <button 
          className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          Chat
        </button>
        <button 
          className={`tab ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          Search Guidelines
        </button>
      </div>

      {activeTab === 'assess' && <AssessmentForm />}
      {activeTab === 'chat' && <ChatInterface />}
      {activeTab === 'search' && <SearchInterface />}
    </div>
  )
}

// Assessment Form Component
function AssessmentForm() {
  const [patients, setPatients] = useState([])
  const [searchTerm, setSearchTerm] = useState('')
  const [showDropdown, setShowDropdown] = useState(false)
  const [form, setForm] = useState({
    patient_id: '',
    age: '',
    sex: '',
    symptoms: '',
    symptom_duration: '',
    risk_factors: '',
    presenting_complaint: ''
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // Load patients on mount
  useEffect(() => {
    api.get('/patients')
      .then(res => setPatients(res.patients || []))
      .catch(err => {
        console.error('Failed to load patients:', err)
        setPatients([])
      })
  }, [])

  // Filter patients by search term
  const filteredPatients = patients.filter(p => 
    p.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (p.name && p.name.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  // Handle patient selection
  const handleSelectPatient = async (patient) => {
    try {
      // Fetch full patient data
      const fullPatient = await api.get(`/patients/${patient.id}`)
      
      setForm({
        patient_id: fullPatient.patient_id,
        age: fullPatient.age ? String(fullPatient.age) : '',
        sex: fullPatient.gender === 'Male' ? 'male' : fullPatient.gender === 'Female' ? 'female' : 'other',
        symptoms: fullPatient.symptoms ? fullPatient.symptoms.join(', ') : '',
        symptom_duration: fullPatient.symptom_duration_days ? String(fullPatient.symptom_duration_days) : '',
        risk_factors: fullPatient.smoking_history ? fullPatient.smoking_history : '',
        presenting_complaint: ''
      })
    } catch (err) {
      console.error('Failed to load patient details:', err)
      setError(`Failed to load patient details: ${err.message}`)
    }
    setSearchTerm('')
    setShowDropdown(false)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await api.post('/assess', {
        patient: {
          patient_id: form.patient_id || null,
          age: form.age ? parseInt(form.age) : null,
          sex: form.sex || null,
          symptoms: form.symptoms.split(',').map(s => s.trim()).filter(Boolean),
          symptom_duration: form.symptom_duration || null,
          risk_factors: form.risk_factors.split(',').map(s => s.trim()).filter(Boolean),
          presenting_complaint: form.presenting_complaint || null
        },
        include_reasoning: true,
        include_differential: true
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleQuickAssess = async () => {
    if (!form.symptoms) {
      setError('Please enter at least one symptom')
      return
    }
    
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await api.post('/assess/quick', {
        symptoms: form.symptoms.split(',').map(s => s.trim()).filter(Boolean),
        patient_id: form.patient_id || null,
        age: form.age ? parseInt(form.age) : null,
        sex: form.sex || null
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="card">
        <h2>Patient Information</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Select Patient</label>
            <div className="searchable-dropdown">
              <input
                type="text"
                placeholder="Search by Patient ID or Name..."
                value={searchTerm}
                onChange={e => {
                  setSearchTerm(e.target.value)
                  setShowDropdown(true)
                }}
                onFocus={() => setShowDropdown(true)}
              />
              {showDropdown && (
                <div className="dropdown-list">
                  {filteredPatients.length > 0 ? (
                    filteredPatients.map(patient => (
                      <div
                        key={patient.id}
                        className="dropdown-item"
                        onClick={() => handleSelectPatient(patient)}
                      >
                        <strong>{patient.id}</strong> - {patient.name} (Age: {patient.age})
                      </div>
                    ))
                  ) : (
                    <div className="dropdown-item">No patients found</div>
                  )}
                </div>
              )}
            </div>
            {form.patient_id && (
              <div className="selected-patient">
                Selected: {form.patient_id}
              </div>
            )}
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Age</label>
              <input
                type="number"
                value={form.age}
                onChange={e => setForm({...form, age: e.target.value})}
                placeholder="e.g., 55"
                disabled={form.patient_id ? true : false}
              />
            </div>
            <div className="form-group">
              <label>Sex</label>
              <select
                value={form.sex}
                onChange={e => setForm({...form, sex: e.target.value})}
                disabled={form.patient_id ? true : false}
              >
                <option value="">Select...</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label>Symptoms (comma-separated)</label>
            <textarea
              value={form.symptoms}
              onChange={e => setForm({...form, symptoms: e.target.value})}
              placeholder="e.g., persistent cough, weight loss, fatigue"
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Symptom Duration (days)</label>
              <input
                type="number"
                value={form.symptom_duration}
                onChange={e => setForm({...form, symptom_duration: e.target.value})}
                placeholder="e.g., 14"
              />
            </div>
            <div className="form-group">
              <label>Risk Factors</label>
              <input
                type="text"
                value={form.risk_factors}
                onChange={e => setForm({...form, risk_factors: e.target.value})}
                placeholder="e.g., Smoking, Family History"
              />
            </div>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? <><span className="spinner"></span> Assessing...</> : 'Full Assessment'}
            </button>
            <button type="button" className="btn btn-secondary" onClick={handleQuickAssess} disabled={loading}>
              Quick Assessment
            </button>
          </div>
        </form>
      </div>

      {error && <div className="error">{error}</div>}
      
      {result && <AssessmentResult result={result} />}
    </div>
  )
}

// Assessment Result Component
function AssessmentResult({ result }) {
  return (
    <div className={`result-box ${result.risk_level}`}>
      <div className="result-header">
        <h3>Assessment Result</h3>
        <span className={`risk-badge ${result.risk_level}`}>
          {result.risk_level.toUpperCase()}
        </span>
      </div>

      <p><strong>Urgency:</strong> {result.urgency_display || result.urgency}</p>
      <p><strong>Summary:</strong> {result.summary}</p>

      {result.recommended_actions?.length > 0 && (
        <>
          <h3>Recommended Actions</h3>
          <ul className="result-list">
            {result.recommended_actions.map((action, i) => (
              <li key={i}>{action}</li>
            ))}
          </ul>
        </>
      )}

      {result.investigations?.length > 0 && (
        <>
          <h3>Investigations</h3>
          <ul className="result-list">
            {result.investigations.map((inv, i) => (
              <li key={i}>{inv}</li>
            ))}
          </ul>
        </>
      )}

      {result.red_flags?.length > 0 && (
        <>
          <h3>⚠️ Red Flags</h3>
          <ul className="result-list">
            {result.red_flags.map((flag, i) => (
              <li key={i}>{flag}</li>
            ))}
          </ul>
        </>
      )}

      {result.reasoning && (
        <>
          <h3>Clinical Reasoning</h3>
          <p>{result.reasoning}</p>
        </>
      )}

      {result.citations?.length > 0 && (
        <div className="citations">
          <strong>References:</strong> {result.citations.join(' | ')}
        </div>
      )}

      <p style={{ marginTop: '15px', fontSize: '0.85rem', color: '#666' }}>
        Confidence: {(result.confidence * 100).toFixed(0)}% | 
        Assessment ID: {result.assessment_id}
      </p>
    </div>
  )
}

// Chat Interface Component
function ChatInterface() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const startChat = async () => {
    setLoading(true)
    try {
      const data = await api.post('/chat/start', {})
      setSessionId(data.session_id)
      setMessages([{ role: 'assistant', content: data.message }])
      setResult(null)
    } catch (err) {
      setMessages([{ role: 'assistant', content: 'Failed to start chat. Is the API running?' }])
    } finally {
      setLoading(false)
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    try {
      const data = await api.post('/chat', {
        session_id: sessionId,
        message: userMessage
      })
      
      setSessionId(data.session_id)
      setMessages(prev => [...prev, { role: 'assistant', content: data.message }])
      
      if (data.assessment_complete && data.assessment) {
        setResult(data.assessment)
      }
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error: ' + err.message }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div>
      <div className="card">
        <h2>Chat Assessment</h2>
        
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <p style={{ marginBottom: '20px', color: '#666' }}>
              Start a conversation to assess a patient through guided questions.
            </p>
            <button className="btn btn-primary" onClick={startChat} disabled={loading}>
              {loading ? 'Starting...' : 'Start New Chat'}
            </button>
          </div>
        ) : (
          <div className="chat-container">
            <div className="chat-messages">
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="message-label">
                    {msg.role === 'user' ? 'You' : 'Assistant'}
                  </div>
                  <div className="message-content">
                    {msg.content}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="message assistant">
                  <div className="message-content">
                    <span className="spinner"></span> Thinking...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="chat-input">
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                disabled={loading}
              />
              <button className="btn btn-primary" onClick={sendMessage} disabled={loading || !input.trim()}>
                Send
              </button>
            </div>

            <div style={{ marginTop: '10px' }}>
              <button className="btn btn-secondary" onClick={startChat} disabled={loading}>
                New Chat
              </button>
            </div>
          </div>
        )}
      </div>

      {result && <AssessmentResult result={result} />}
    </div>
  )
}

// Search Interface Component
function SearchInterface() {
  const [query, setQuery] = useState('')
  const [cancerType, setCancerType] = useState('')
  const [urgentOnly, setUrgentOnly] = useState(false)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const data = await api.post('/search', {
        query: query.trim(),
        top_k: 5,
        cancer_type: cancerType || null,
        urgent_only: urgentOnly
      })
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="card">
        <h2>Search NG12 Guidelines</h2>
        <form onSubmit={handleSearch}>
          <div className="form-group">
            <label>Search Query</label>
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="e.g., lung cancer symptoms, haemoptysis referral"
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Cancer Type (optional)</label>
              <select value={cancerType} onChange={e => setCancerType(e.target.value)}>
                <option value="">All types</option>
                <option value="lung">Lung</option>
                <option value="breast">Breast</option>
                <option value="colorectal">Colorectal</option>
                <option value="prostate">Prostate</option>
                <option value="skin">Skin/Melanoma</option>
                <option value="bladder">Bladder</option>
                <option value="lymphoma">Lymphoma</option>
              </select>
            </div>
            <div className="form-group">
              <label>&nbsp;</label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'normal' }}>
                <input
                  type="checkbox"
                  checked={urgentOnly}
                  onChange={e => setUrgentOnly(e.target.checked)}
                />
                Urgent recommendations only
              </label>
            </div>
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading || !query.trim()}>
            {loading ? <><span className="spinner"></span> Searching...</> : 'Search'}
          </button>
        </form>
      </div>

      {error && <div className="error">{error}</div>}

      {results && (
        <div className="card">
          <h2>Results ({results.total_results})</h2>
          {results.expanded_query && (
            <p style={{ fontSize: '0.85rem', color: '#666', marginBottom: '15px' }}>
              Expanded query: {results.expanded_query}
            </p>
          )}

          {results.results.length === 0 ? (
            <p>No results found.</p>
          ) : (
            results.results.map((r, i) => (
              <div key={i} className="search-result">
                <div className="search-result-header">
                  <span>{r.citation || `Page ${r.page}`}</span>
                  <span className="score-badge">Score: {(r.score * 100).toFixed(0)}%</span>
                </div>
                <div className="search-result-text">{r.text}</div>
                {r.clinical_metadata?.urgency && (
                  <div style={{ marginTop: '8px', fontSize: '0.85rem' }}>
                    <strong>Urgency:</strong> {r.clinical_metadata.urgency}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
