import React from 'react'
import { X } from 'lucide-react'
import './SettingsModal.css'

export function SettingsModal({ 
  isOpen, 
  onClose, 
  useRag, setUseRag,
  historyLimit, setHistoryLimit,
  maxTokens, setMaxTokens,
  temperature, setTemperature,
  repeatPenalty, setRepeatPenalty,
  preferredLanguage, setPreferredLanguage,
  systemPrompt, setSystemPrompt,
  adminKey, setAdminKey,
  checkpointSize, setCheckpointSize
}) {
  if (!isOpen) return null

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={e => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="close-btn" onClick={onClose}><X size={20} /></button>
        </div>
        
        <div className="settings-content">
          <div className="setting-group">
            <label className="rag-toggle">
              <input 
                type="checkbox" 
                checked={useRag} 
                onChange={(e) => setUseRag(e.target.checked)} 
              />
              <span>Use RAG Memory</span>
            </label>
            <p className="setting-desc">Allows the AI to search your codebase for context.</p>
          </div>

          <div className="setting-group">
            <div className="setting-label">
              <label>Context Size</label>
              <span className="value-badge">{historyLimit} msgs</span>
            </div>
            <input 
              type="range" 
              min="1" 
              max="20" 
              value={historyLimit} 
              onChange={(e) => setHistoryLimit(parseInt(e.target.value))} 
            />
          </div>

          <div className="setting-group">
            <div className="setting-label">
              <label>Max Tokens</label>
              <span className="value-badge">{maxTokens}</span>
            </div>
            <input 
              type="range" 
              min="512" 
              max="16384" 
              step="256"
              value={maxTokens} 
              onChange={(e) => setMaxTokens(parseInt(e.target.value))} 
            />
          </div>

          <div className="setting-group">
            <div className="setting-label">
              <label>Temperature</label>
              <span className="value-badge">{temperature}</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="1.5" 
              step="0.1"
              value={temperature} 
              onChange={(e) => setTemperature(parseFloat(e.target.value))} 
            />
            <p className="setting-desc">Higher values = more creative/random. Lower = more focused.</p>
          </div>

          <div className="setting-group">
            <div className="setting-label">
              <label>Repetition Penalty</label>
              <span className="value-badge">{repeatPenalty}</span>
            </div>
            <input 
              type="range" 
              min="1.0" 
              max="2.0" 
              step="0.05"
              value={repeatPenalty} 
              onChange={(e) => setRepeatPenalty(parseFloat(e.target.value))} 
            />
          </div>

          <div className="setting-group">
            <label>Admin Secret Key</label>
            <input 
              type="password" 
              className="text-input"
              value={adminKey || ''} 
              onChange={(e) => setAdminKey(e.target.value)} 
              placeholder="Enter secret key for model control"
            />
            <p className="setting-desc">Required to load/unload models.</p>
          </div>

          {adminKey && (
            <div className="setting-group">
              <div className="setting-label">
                <label>Summary Checkpoint Interval</label>
                <span className="value-badge">{checkpointSize} msgs</span>
              </div>
              <input
                type="range"
                min="3"
                max="200"
                value={checkpointSize}
                onChange={(e) => setCheckpointSize(parseInt(e.target.value))}
              />
              <p className="setting-desc">Admin-only: how many messages between summary checkpoints.</p>
            </div>
          )}

          <div className="setting-group">
            <label>Preferred Language</label>
            <select 
              className="language-select"
              value={preferredLanguage}
              onChange={(e) => setPreferredLanguage(e.target.value)}
            >
              <option value="ru">🇷🇺 Русский</option>
              <option value="en">🇬🇧 English</option>
              <option value="zh">🇨🇳 中文</option>
              <option value="es">🇪🇸 Español</option>
              <option value="auto">🌐 Auto (Model Choice)</option>
            </select>
            <p className="setting-desc">Language for AI responses. Ignored if custom system prompt is set.</p>
          </div>

          <div className="setting-group">
            <label>System Prompt (Optional)</label>
            <textarea 
              className="system-prompt-input"
              placeholder="Override system prompt (e.g., 'You are a pirate')..."
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={5}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
