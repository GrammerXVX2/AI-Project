import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, Paperclip, MessageSquare, Plus, Square, Trash2, Settings, PanelLeftClose, PanelLeftOpen, Image as ImageIcon } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { SystemMonitor } from './components/SystemMonitor'
import { SettingsModal } from './components/SettingsModal'
import { MessageContent } from './components/MessageContent'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Привет! Я ваш AI-помощник. Чем могу помочь с кодом сегодня?' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [mode, setMode] = useState('chat') // 'chat' | 'image'
  const [imgParams, setImgParams] = useState({
    width: 1024,
    height: 1024,
    steps: 9,
    seed: 42,
    enhance_prompt: true,
    low_vram: false,
    offload_cpu: true
  })
  
  // Helper for lazy state initialization
  const getStoredSetting = (key, defaultValue) => {
    try {
      const savedSettings = localStorage.getItem('ai-chat-settings')
      if (savedSettings) {
        const settings = JSON.parse(savedSettings)
        return settings[key] !== undefined ? settings[key] : defaultValue
      }
    } catch (e) {
      console.error('Error reading settings', e)
    }
    return defaultValue
  }

  // Settings State with Lazy Initialization
  const [useRag, setUseRag] = useState(() => getStoredSetting('useRag', true))
  const [historyLimit, setHistoryLimit] = useState(() => getStoredSetting('historyLimit', 5))
  const [maxTokens, setMaxTokens] = useState(() => getStoredSetting('maxTokens', 512))
  const [temperature, setTemperature] = useState(() => getStoredSetting('temperature', 0.7))
  const [repeatPenalty, setRepeatPenalty] = useState(() => getStoredSetting('repeatPenalty', 1.1))
  const [preferredLanguage, setPreferredLanguage] = useState(() => getStoredSetting('preferredLanguage', 'ru'))
  const [systemPrompt, setSystemPrompt] = useState(() => getStoredSetting('systemPrompt', ''))
  const [adminKey, setAdminKey] = useState(() => getStoredSetting('adminKey', ''))
  const [checkpointSize, setCheckpointSize] = useState(() => getStoredSetting('checkpointSize', 25))
  const [mySessionIds, setMySessionIds] = useState(() => {
    try {
      const saved = localStorage.getItem('my-session-ids')
      return saved ? JSON.parse(saved) : []
    } catch { return [] }
  })
  
  const [sessionId, setSessionId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [isSummaryOpen, setIsSummaryOpen] = useState(false)
  const [summaries, setSummaries] = useState([])
  const [expandedSummaries, setExpandedSummaries] = useState({})
  const [expandedCheckpoints, setExpandedCheckpoints] = useState({})
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)
  const abortControllerRef = useRef(null)
  const progressIntervalRef = useRef(null)
  const [imageProgress, setImageProgress] = useState(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  // We no longer poll fine-grained progress to avoid noisy logs.

  // Загрузка списка сессий при старте
  useEffect(() => {
    fetchSessions()
  }, [])

  // Сохранение настроек в localStorage при изменении
  useEffect(() => {
    const settings = {
      useRag,
      historyLimit,
      maxTokens,
      temperature,
      repeatPenalty,
      preferredLanguage,
      systemPrompt,
      adminKey,
      checkpointSize,
    }
    localStorage.setItem('ai-chat-settings', JSON.stringify(settings))
  }, [useRag, historyLimit, maxTokens, temperature, repeatPenalty, preferredLanguage, systemPrompt, adminKey, checkpointSize])

  // Save mySessionIds
  useEffect(() => {
    localStorage.setItem('my-session-ids', JSON.stringify(mySessionIds))
  }, [mySessionIds])

  const fetchSessions = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/sessions')
      if (res.ok) {
        const data = await res.json()
        // Filter sessions to only show ones created by this client
        const mySessions = data.filter(s => mySessionIds.includes(s.id))
        setSessions(mySessions)
      }
    } catch (e) {
      console.error("Failed to fetch sessions", e)
    }
  }

  const fetchSummaries = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/sessions/summary')
      if (res.ok) {
        const data = await res.json()
        setSummaries(data)
      }
    } catch (e) {
      console.error('Failed to fetch summaries', e)
    }
  }

  const toggleSummaryExpanded = (id) => {
    setExpandedSummaries(prev => ({ ...prev, [id]: !prev[id] }))
  }

  const toggleCheckpoints = (id) => {
    setExpandedCheckpoints(prev => ({ ...prev, [id]: !prev[id] }))
  }

  const openSummaries = () => {
    if (!adminKey) return
    fetchSummaries()
    setIsSummaryOpen(true)
  }

  const deleteSession = async (e, id) => {
    e.stopPropagation() // Prevent loading the session when clicking delete
    if (!confirm('Are you sure you want to delete this chat?')) return

    try {
      const res = await fetch(`http://localhost:8000/api/history/${id}`, {
        method: 'DELETE'
      })
      
      if (res.ok) {
        // If we deleted the current session, start a new one
        if (id === sessionId) {
          startNewChat()
        }
        // Remove from local list
        setMySessionIds(prev => prev.filter(sid => sid !== id))
        fetchSessions()
      }
    } catch (e) {
      console.error("Failed to delete session", e)
    }
  }

  const loadSession = async (id) => {
    if (id === sessionId) return
    setIsLoading(true)
    try {
      const res = await fetch(`http://localhost:8000/api/history/${id}`)
      if (res.ok) {
        const history = await res.json()
        setMessages(history.length ? history : [])
        setSessionId(id)
      }
    } catch (e) {
      console.error("Failed to load session", e)
    } finally {
      setIsLoading(false)
    }
  }

  const startNewChat = () => {
    setSessionId(null)
    setMessages([{ role: 'assistant', content: 'Привет! Я ваш AI-помощник. Чем могу помочь с кодом сегодня?' }])
  }

  const stopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
      setIsLoading(false)
    }
  }

  const handleMessageContentUpdate = async (messageIndex, newContent) => {
    setMessages(prev => prev.map((m, idx) => idx === messageIndex ? { ...m, content: newContent } : m))
    if (!sessionId) return
    try {
      await fetch(`http://localhost:8000/api/history/${sessionId}/message/${messageIndex}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: newContent })
      })
    } catch (e) {
      console.error('Failed to persist message update', e)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    if (mode === 'image') {
      try {
        setMessages(prev => [...prev, { role: 'assistant', content: 'Generating image...', streaming: true, meta: { image_pending: true } }])

        const reqSessionId = sessionId || (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2))
        if (!sessionId) {
          setSessionId(reqSessionId)
          setMySessionIds(prev => {
            if (!prev.includes(reqSessionId)) {
              return [reqSessionId, ...prev]
            }
            return prev
          })
        }
        
        // No progress polling to reduce spam in logs; rely on placeholder spinner.

        const res = await fetch('http://localhost:8000/api/image/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt: userMessage.content,
            session_id: reqSessionId,
            ...imgParams
          })
        })
        
        if (!res.ok) throw new Error('Image generation failed')
        
        const data = await res.json()

        if (data.session_id && !sessionId) {
          setSessionId(data.session_id)
          setMySessionIds(prev => {
            if (!prev.includes(data.session_id)) {
              return [data.session_id, ...prev]
            }
            return prev
          })
        }
        
        setMessages(prev => {
          const newMessages = [...prev]
          let imageUrl = data.image_file || data.image
          const duration = data.duration
          if (imageUrl && imageUrl.startsWith('/')) {
            imageUrl = `http://localhost:8000${imageUrl}`
          }
          newMessages[newMessages.length - 1] = {
            role: 'assistant',
            content: `![Generated Image](${imageUrl})\n\n*Prompt: ${data.prompt_used}*`,
            meta: { model: 'Z-Image-Turbo', image_url: imageUrl, duration }
          }
          return newMessages
        })
        setImageProgress(null)
      } catch (e) {
        console.error(e)
        setMessages(prev => {
          const newMessages = [...prev]
          newMessages[newMessages.length - 1] = {
            role: 'assistant',
            content: `Error generating image: ${e.message}`
          }
          return newMessages
        })
      } finally {
        setIsLoading(false)
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current)
          progressIntervalRef.current = null
        }
      }
      return
    }

    // Create new AbortController
    abortControllerRef.current = new AbortController()

    // Формируем системный промпт с учетом языка
    let finalSystemPrompt = systemPrompt
    if (!systemPrompt.trim()) {
      const languageInstructions = {
        'ru': 'Always respond in Russian language.',
        'en': 'Always respond in English language.',
        'zh': 'Always respond in Chinese language.',
        'es': 'Always respond in Spanish language.',
        'auto': ''
      }
      finalSystemPrompt = languageInstructions[preferredLanguage] || ''
    }

    // Создаем пустое сообщение-заглушку для streaming
    const streamingMessageIndex = messages.length + 1
    setMessages(prev => [...prev, { role: 'assistant', content: '', streaming: true }])

    try {
      const payload = {
        prompt: userMessage.content,
        use_rag: useRag,
        history_limit: historyLimit,
        max_tokens: maxTokens,
        temperature: temperature,
        repeat_penalty: repeatPenalty,
        system_prompt: finalSystemPrompt,
        session_id: sessionId
      }

      if (adminKey) {
        payload.admin_key = adminKey
        payload.summary_checkpoint_size = checkpointSize
      }

      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) throw new Error('Network response was not ok')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let fullContent = ''
      let metadata = {}

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.type === 'metadata') {
                metadata = data
                if (!sessionId && data.session_id) {
                  setSessionId(data.session_id)
                  // Add new session ID to our local list
                  setMySessionIds(prev => {
                    if (!prev.includes(data.session_id)) {
                      return [data.session_id, ...prev]
                    }
                    return prev
                  })
                  fetchSessions()
                }
              } else if (data.type === 'content') {
                fullContent += data.content
                // Обновляем сообщение с накопленным контентом
                setMessages(prev => {
                  const newMessages = [...prev]
                  newMessages[streamingMessageIndex] = {
                    role: 'assistant',
                    content: fullContent,
                    streaming: true,
                    meta: {
                      category: metadata.category,
                      rag_used: metadata.rag_used,
                      model: metadata.model
                    }
                  }
                  return newMessages
                })
              } else if (data.type === 'done') {
                // Завершаем streaming
                setMessages(prev => {
                  const newMessages = [...prev]
                  delete newMessages[streamingMessageIndex].streaming
                  return newMessages
                })
                if (sessionId) fetchSessions()
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e)
            }
          }
        }
      }

    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Generation stopped by user')
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: 'Generation stopped.' 
        }])
      } else {
        console.error('Error:', error)
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: 'Извините, произошла ошибка при соединении с сервером.' 
        }])
      }
    } finally {
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className={`sidebar ${!isSidebarOpen ? 'collapsed' : ''}`}>
        <div className="new-chat-btn" onClick={startNewChat}>
          <Plus size={16} />
          <span>New Chat</span>
        </div>
        <div className="history">
          {sessions.map(session => (
            <div 
              key={session.id} 
              className={`history-item ${sessionId === session.id ? 'active' : ''}`}
              onClick={() => loadSession(session.id)}
            >
              <MessageSquare size={14} />
              <span className="history-title">{session.title || "New Chat"}</span>
              <button 
                className="delete-chat-btn"
                onClick={(e) => deleteSession(e, session.id)}
                title="Delete chat"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
        <div className="sidebar-footer">
          <button className="settings-btn" onClick={() => setIsSettingsOpen(true)}>
            <Settings size={16} />
            <span>Settings</span>
          </button>
          {adminKey && (
            <button className="settings-btn" onClick={openSummaries}>
              <Settings size={16} />
              <span>Summaries</span>
            </button>
          )}
          <SystemMonitor adminKey={adminKey} />
        </div>
      </div>

      <SettingsModal 
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        useRag={useRag} setUseRag={setUseRag}
        historyLimit={historyLimit} setHistoryLimit={setHistoryLimit}
        maxTokens={maxTokens} setMaxTokens={setMaxTokens}
        temperature={temperature} setTemperature={setTemperature}
        repeatPenalty={repeatPenalty} setRepeatPenalty={setRepeatPenalty}
        preferredLanguage={preferredLanguage} setPreferredLanguage={setPreferredLanguage}
        systemPrompt={systemPrompt} setSystemPrompt={setSystemPrompt}
        adminKey={adminKey} setAdminKey={setAdminKey}
        checkpointSize={checkpointSize} setCheckpointSize={setCheckpointSize}
      />

      {isSummaryOpen && (
        <div className="settings-modal-overlay" onClick={() => setIsSummaryOpen(false)}>
          <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
            <div className="settings-header">
              <h2>Сводки чатов (admin)</h2>
              <button className="close-btn" onClick={() => setIsSummaryOpen(false)}>✕</button>
            </div>
            <div className="settings-content">
              {summaries.length === 0 && <div style={{ color: '#aaa' }}>Нет данных</div>}
              {summaries.map(item => (
                <div key={item.id} style={{ border: '1px solid #333', borderRadius: '6px', padding: '0.75rem', background: '#161616' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', marginBottom: '0.4rem' }}>
                    <strong style={{ color: '#e5e7eb' }}>{item.title || 'Без названия'}</strong>
                    <span style={{ color: '#888', fontSize: '0.8rem' }}>{item.date}</span>
                  </div>
                  {item.summary_checkpoints && item.summary_checkpoints.length > 0 && (
                    <div style={{ marginBottom: '0.5rem' }}>
                      <button
                        style={{
                          background: '#111827',
                          border: '1px solid #334155',
                          color: '#e2e8f0',
                          borderRadius: '6px',
                          padding: '4px 8px',
                          cursor: 'pointer'
                        }}
                        onClick={() => toggleCheckpoints(item.id)}
                      >
                        Чекпоинты: {item.summary_checkpoints.length} {expandedCheckpoints[item.id] ? '▲' : '▼'}
                      </button>
                      {expandedCheckpoints[item.id] && (
                        <div style={{ marginTop: '0.35rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                          {item.summary_checkpoints.map((cp, idx) => (
                            <div key={idx} style={{ border: '1px solid #1f2937', borderRadius: '6px', padding: '0.5rem', background: '#0f172a' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.8rem', marginBottom: '0.25rem' }}>
                                <span>Диапазон: {cp.range}</span>
                                {cp.created_at && <span>{cp.created_at}</span>}
                              </div>
                              <div style={{ color: '#cbd5e1', whiteSpace: 'pre-wrap', fontSize: '0.9rem' }}>
                                {cp.summary || '—'}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                  <div style={{ color: '#cbd5e1', whiteSpace: 'pre-wrap' }}>
                    {expandedSummaries[item.id]
                      ? (item.summary || 'Нет выжимки')
                      : ((item.summary || 'Нет выжимки').split(' ').slice(0, 10).join(' ') + ((item.summary || '').split(' ').length > 10 ? '…' : ''))}
                  </div>
                  {(item.summary || '').split(' ').length > 10 && (
                    <button
                      style={{ marginTop: '0.35rem', background: '#1f2937', border: '1px solid #334155', color: '#e2e8f0', borderRadius: '6px', padding: '4px 8px', cursor: 'pointer' }}
                      onClick={() => toggleSummaryExpanded(item.id)}
                    >
                      {expandedSummaries[item.id] ? 'Свернуть' : 'Развернуть'}
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="chat-area">
        <div className="chat-header">
          <button 
            className="sidebar-toggle" 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            title={isSidebarOpen ? "Close Sidebar" : "Open Sidebar"}
          >
            {isSidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
          </button>
        </div>
        <div className="messages-container">
          {messages.map((msg, index) => (
            <div key={index} className={`message-wrapper ${msg.role}`}>
              <div className="message-content">
                <div className="avatar">
                  {msg.role === 'assistant' ? <Bot size={24} /> : <User size={24} />}
                </div>
                <div className="text-content">
                  <div className="sender-name">
                    {msg.role === 'assistant' ? 'AI Assistant' : 'You'}
                    {msg.meta && (
                        <span className="meta-badge">
                          {msg.meta.model || msg.meta.category}
                          {msg.meta.duration ? ` · ${Math.round(msg.meta.duration * 10) / 10}s` : ''}
                          {msg.meta.rag_used ? ' (RAG)' : ''}
                        </span>
                      )}
                  </div>
                  {msg.role === 'assistant' ? (
                    msg.meta && msg.meta.image_pending ? (
                      <div className="image-placeholder">
                        <Loader2 className="spin" size={24} />
                        <span>Generating image...</span>
                      </div>
                    ) : (
                      <>
                        <MessageContent
                          content={msg.content}
                          streaming={!!msg.streaming}
                          adminMode={!!adminKey}
                          onContentUpdate={(updated) => handleMessageContentUpdate(index, updated)}
                        />
                        {msg.meta && msg.meta.image_url && (
                          <div className="image-actions">
                            <button className="image-btn disabled" disabled>
                              Copy URL
                            </button>
                            <a className="image-btn" href={msg.meta.image_url} download target="_blank" rel="noreferrer">
                              Download
                            </a>
                          </div>
                        )}
                      </>
                    )
                  ) : (
                    <div className="markdown">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          {isLoading && mode === 'chat' && (
            <div className="message-wrapper assistant">
              <div className="message-content">
                <div className="avatar"><Bot size={24} /></div>
                <div className="text-content">
                  <div className="loading-indicator">
                    <Loader2 className="spin" size={20} />
                    <span>Thinking...</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-area">
          <div className="input-container">
            <div className="mode-switcher">
              <button 
                className={`mode-btn ${mode === 'chat' ? 'active' : ''}`}
                onClick={() => setMode('chat')}
              >
                <MessageSquare size={16} /> Chat
              </button>
              <button 
                className={`mode-btn ${mode === 'image' ? 'active' : ''}`}
                onClick={() => setMode('image')}
              >
                <ImageIcon size={16} /> Image
              </button>
            </div>

            {mode === 'image' && (
              <div className="image-params">
                <div className="param-group">
                  <label>Size</label>
                  <select 
                    value={`${imgParams.width}x${imgParams.height}`}
                    onChange={(e) => {
                      const [w, h] = e.target.value.split('x').map(Number)
                      setImgParams(p => ({...p, width: w, height: h}))
                    }}
                  >
                    <option value="512x512">512x512</option>
                    <option value="768x768">768x768</option>
                    <option value="1024x1024">1024x1024</option>
                    <option value="1280x720">1280x720 (16:9)</option>
                  </select>
                </div>
                <div className="param-group">
                  <label>Steps</label>
                  <input 
                    type="number" 
                    min="1" max="50" 
                    value={imgParams.steps}
                    onChange={(e) => setImgParams(p => ({...p, steps: parseInt(e.target.value)}))}
                  />
                </div>
                <div className="param-group">
                  <label>Seed</label>
                  <input 
                    type="number" 
                    value={imgParams.seed}
                    onChange={(e) => setImgParams(p => ({...p, seed: parseInt(e.target.value)}))}
                  />
                </div>
                <label className="checkbox-param">
                  <input 
                    type="checkbox"
                    checked={imgParams.enhance_prompt}
                    onChange={(e) => setImgParams(p => ({...p, enhance_prompt: e.target.checked}))}
                  />
                  Enhance
                </label>
                <label className="checkbox-param">
                  <input 
                    type="checkbox"
                    checked={imgParams.low_vram}
                    onChange={(e) => setImgParams(p => ({...p, low_vram: e.target.checked}))}
                  />
                  Low VRAM
                </label>
                <label className="checkbox-param">
                  <input 
                    type="checkbox"
                    checked={imgParams.offload_cpu}
                    onChange={(e) => setImgParams(p => ({...p, offload_cpu: e.target.checked}))}
                  />
                  CPU Offload
                </label>
              </div>
            )}

            <form onSubmit={handleSubmit}>
              <div className="input-wrapper">
                <button type="button" className="attach-btn">
                  <Paperclip size={20} />
                </button>
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Message AI..."
                  disabled={isLoading}
                  rows={1}
                />
                {isLoading ? (
                  <button 
                    type="button" 
                    className="stop-btn"
                    onClick={stopGeneration}
                  >
                    <Square size={14} fill="currentColor" />
                  </button>
                ) : (
                  <button 
                    type="submit" 
                    className={`send-btn ${input.trim() ? 'active' : ''}`}
                    disabled={!input.trim()}
                  >
                    <Send size={18} />
                  </button>
                )}
              </div>
            </form>
            <div className="disclaimer">
              AI can make mistakes. Consider checking important information.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
