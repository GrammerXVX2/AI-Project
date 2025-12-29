import { useState, useEffect } from 'react'
import { Cpu, Database, Activity, Thermometer, ChevronDown, ChevronUp } from 'lucide-react'
import './SystemMonitor.css'

export function SystemMonitor({ adminKey }) {
  const [stats, setStats] = useState(null)
  const [loadModes, setLoadModes] = useState({})
  const [isCollapsed, setIsCollapsed] = useState(false)

  const isAdmin = !!adminKey && adminKey.length > 0;

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stats')

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setStats(data)
    }

    return () => {
      ws.close()
    }
  }, [])

  const toggleModel = async (name, currentStatus) => {
    const action = currentStatus === 'unloaded' ? 'load' : 'unload';
    const mode = loadModes[name] || 'gpu';
    try {
      let url = `http://localhost:8000/api/models/${name}/${action}`;
      if (action === 'load') {
        url += `?mode=${mode}`;
      }
      await fetch(url, { method: 'POST' });
    } catch (e) {
      console.error("Failed to toggle model", e);
    }
  };

  if (!stats) return <div className="system-monitor loading">Connecting to system...</div>

  const gpu = stats.gpu_stats

  return (
    <div className={`system-monitor ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="monitor-header" onClick={() => setIsCollapsed(!isCollapsed)}>
        <span>System Status</span>
        {isCollapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </div>

      <div className={`monitor-content ${isCollapsed ? 'collapsed' : ''}`}>
        <div className="monitor-section">
          <div className="monitor-row">
            <div className="monitor-label">
              <Cpu size={14} />
              <span>CPU</span>
            </div>
            <div className="monitor-bar-container">
              <div className="monitor-bar" style={{ width: `${stats.cpu_percent}%`, backgroundColor: getColor(stats.cpu_percent) }}></div>
            </div>
            <div className="monitor-text">{stats.cpu_percent}%</div>
          </div>

          <div className="monitor-row">
            <div className="monitor-label">
              <Database size={14} />
              <span>RAM</span>
            </div>
            <div className="monitor-bar-container">
              <div className="monitor-bar" style={{ width: `${stats.ram_percent}%`, backgroundColor: getColor(stats.ram_percent) }}></div>
            </div>
            <div className="monitor-text">{stats.ram_used_gb}GB</div>
          </div>

          {gpu && (
            <>
              <div className="monitor-row">
                <div className="monitor-label">
                  <Activity size={14} />
                  <span>GPU</span>
                </div>
                <div className="monitor-bar-container">
                  <div className="monitor-bar" style={{ width: `${gpu.load}%`, backgroundColor: getColor(gpu.load) }}></div>
                </div>
                <div className="monitor-text">{gpu.load}%</div>
              </div>
              
              <div className="monitor-row">
                <div className="monitor-label">
                  <span style={{marginLeft: 18}}>VRAM</span>
                </div>
                <div className="monitor-bar-container">
                  <div className="monitor-bar" style={{ width: `${gpu.memory_percent}%`, backgroundColor: getColor(gpu.memory_percent) }}></div>
                </div>
                <div className="monitor-text">{gpu.memory_used_gb}GB</div>
              </div>

              {gpu.temp && (
                <div className="monitor-row temp-row">
                    <Thermometer size={14} />
                    <span>{gpu.temp}°C</span>
                </div>
              )}
            </>
          )}
        </div>

        {stats.models && (
          <div className="models-section">
            <div className="section-title">Active Models</div>
            {Object.entries(stats.models).map(([name, info]) => (
              <div key={name} className={`model-row ${info.status === 'generating' ? 'active' : ''}`}>
                <div className="model-info">
                  <span className="model-name">{name}</span>
                  <span className="model-meta">{info.config.type} • {info.config.n_ctx}ctx</span>
                </div>
                <div className="model-actions">
                    {name !== 'orchestrator' && isAdmin && (
                    <div className="model-controls">
                      {info.status === 'unloaded' && (
                        <select 
                          className="mode-select"
                          value={loadModes[name] || 'gpu'}
                          onChange={(e) => setLoadModes(prev => ({...prev, [name]: e.target.value}))}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <option value="gpu">GPU</option>
                          <option value="hybrid">50/50</option>
                          <option value="cpu">CPU</option>
                        </select>
                      )}
                      <button 
                          className={`toggle-btn ${info.status === 'unloaded' ? 'load' : 'unload'}`}
                          onClick={() => toggleModel(name, info.status)}
                          title={info.status === 'unloaded' ? 'Load Model' : 'Unload Model'}
                      >
                          {info.status === 'unloaded' ? 'Load' : 'Unload'}
                      </button>
                    </div>
                  )}
                  <div className="model-status">
                      {info.status === 'generating' ? (
                      <span className="status-badge active">Gen</span>
                      ) : info.status === 'unloaded' ? (
                      <span className="status-badge unloaded">Off</span>
                      ) : (
                      <span className="status-badge idle">Idle</span>
                      )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function getColor(percent) {
  if (percent < 50) return '#10a37f'; // Green
  if (percent < 80) return '#f5a623'; // Orange
  return '#ef4444'; // Red
}
