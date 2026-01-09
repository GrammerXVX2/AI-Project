import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { ChevronDown, ChevronRight, Copy, Check } from 'lucide-react'
import mermaid from 'mermaid'
import './MessageContent.css'

mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'dark' })

function MermaidBlock({ chart, adminMode = false, onCodeChange }) {
  const [chartCode, setChartCode] = useState(chart)
  const [draftCode, setDraftCode] = useState(chart)
  const [showCode, setShowCode] = useState(false)
  const [svg, setSvg] = useState('')
  const [error, setError] = useState(null)
  const renderId = useRef(`mermaid-${Math.random().toString(36).slice(2)}`)
  const containerRef = useRef(null)
  const [viewBox, setViewBox] = useState(null)
  const baseBoxRef = useRef(null)

  useEffect(() => {
    setChartCode(chart)
    setDraftCode(chart)
    setShowCode(false)
  }, [chart])

  useEffect(() => {
    let cancelled = false

    const extractViewBox = (svgString) => {
      try {
        const doc = new DOMParser().parseFromString(svgString, 'image/svg+xml')
        const el = doc.querySelector('svg')
        if (!el) return null
        const vb = el.getAttribute('viewBox')
        if (vb) {
          const [x, y, w, h] = vb.split(/\s+/).map(Number)
          return { x, y, w, h }
        }
        const wAttr = Number(el.getAttribute('width')) || 1000
        const hAttr = Number(el.getAttribute('height')) || 1000
        return { x: 0, y: 0, w: wAttr, h: hAttr }
      } catch {
        return null
      }
    }

    const renderChart = async () => {
      setError(null)
      setSvg('')
      setViewBox(null)
      baseBoxRef.current = null
      try {
        // Validate syntax first to avoid rendering huge inline error SVGs
        mermaid.parse(chartCode)
        // Render into a hidden offscreen container so Mermaid can measure layout safely
        let renderRoot = document.getElementById('mermaid-render-root')
        if (!renderRoot) {
          renderRoot = document.createElement('div')
          renderRoot.id = 'mermaid-render-root'
          renderRoot.style.position = 'absolute'
          renderRoot.style.left = '-9999px'
          renderRoot.style.top = '-9999px'
          renderRoot.style.width = '0'
          renderRoot.style.height = '0'
          renderRoot.style.overflow = 'hidden'
          renderRoot.style.pointerEvents = 'none'
          document.body.appendChild(renderRoot)
        }
        renderRoot.innerHTML = ''

        const { svg } = await mermaid.render(renderId.current, chartCode, renderRoot)
        renderRoot.innerHTML = ''
        // Mermaid returns an error SVG on failure; filter it out explicitly
        const looksLikeError = /Syntax error in text|Parse error|mermaid version/i.test(svg)
        if (!cancelled) {
          if (looksLikeError) {
            setError('Mermaid render failed: invalid or unsupported syntax')
            setSvg('')
            setViewBox(null)
            baseBoxRef.current = null
            return
          } else {
            setSvg(svg)
            setError(null)
            const baseBox = extractViewBox(svg)
            baseBoxRef.current = baseBox
            setViewBox(baseBox)
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err?.message || 'Failed to render diagram')
          setSvg('')
          setViewBox(null)
          baseBoxRef.current = null
        }
      }
    }

    renderChart()

    return () => {
      cancelled = true
    }
  }, [chartCode])

  useEffect(() => {
    if (!svg || !viewBox) return
    const svgEl = containerRef.current?.querySelector('svg')
    if (svgEl) {
      svgEl.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`)
      svgEl.removeAttribute('width')
      svgEl.removeAttribute('height')
    }
  }, [svg, viewBox])

  const clampScale = (value) => Math.min(5, Math.max(0.4, value))

  const handleWheel = (e) => {
    if (!containerRef.current || !viewBox || !baseBoxRef.current) return
    e.preventDefault()
    e.stopPropagation()
    const rect = containerRef.current.getBoundingClientRect()
    const pointX = e.clientX - rect.left
    const pointY = e.clientY - rect.top
    const delta = e.deltaY
    const base = baseBoxRef.current
    const currentScale = base && viewBox ? base.w / viewBox.w : 1
    const targetScale = clampScale(currentScale * (delta > 0 ? 0.9 : 1.1))
    const nextWidth = base.w / targetScale
    const nextHeight = base.h / targetScale
    const relX = pointX / rect.width
    const relY = pointY / rect.height
    const nextX = viewBox.x + relX * (viewBox.w - nextWidth)
    const nextY = viewBox.y + relY * (viewBox.h - nextHeight)
    setViewBox({ x: nextX, y: nextY, w: nextWidth, h: nextHeight })
  }

  const dragState = useRef({ dragging: false, startX: 0, startY: 0, originX: 0, originY: 0 })

  const handleMouseDown = (e) => {
    if (e.button !== 0) return
    e.preventDefault()
    dragState.current = {
      dragging: true,
      startX: e.clientX,
      startY: e.clientY,
      originX: viewBox?.x || 0,
      originY: viewBox?.y || 0
    }

    const handleMove = (moveEvent) => {
      if (!dragState.current.dragging) return
      if (!containerRef.current || !viewBox || !baseBoxRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const dx = moveEvent.clientX - dragState.current.startX
      const dy = moveEvent.clientY - dragState.current.startY
      const moveX = (dx / rect.width) * viewBox.w
      const moveY = (dy / rect.height) * viewBox.h
      setViewBox((prev) => ({ ...prev, x: dragState.current.originX - moveX, y: dragState.current.originY - moveY }))
    }

    const handleUp = () => {
      dragState.current.dragging = false
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
    }

    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)
  }

  const handleReset = () => {
    if (baseBoxRef.current) setViewBox(baseBoxRef.current)
  }

  const getCurrentSvgMarkup = () => {
    const svgEl = containerRef.current?.querySelector('svg')
    if (!svgEl) return svg
    const clone = svgEl.cloneNode(true)
    return new XMLSerializer().serializeToString(clone)
  }

  const downloadFile = (data, filename, type) => {
    const blob = new Blob([data], { type })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const handleCopyCode = async () => {
    try {
      await navigator.clipboard.writeText(showCode ? draftCode : chartCode)
    } catch (err) {
      console.error('Copy failed', err)
    }
  }

  const handleApplyDraft = () => {
    try {
      mermaid.parse(draftCode)
    } catch (e) {
      setError(e?.message || 'Failed to parse diagram')
      return
    }
    const prevCode = chartCode
    setChartCode(draftCode)
    setShowCode(false)
    onCodeChange?.(draftCode, prevCode)
  }

  const handleDownloadSvg = () => {
    const svgContent = getCurrentSvgMarkup()
    if (!svgContent) return
    downloadFile(svgContent, 'diagram.svg', 'image/svg+xml')
  }

  const handleDownloadXml = () => {
    const svgContent = getCurrentSvgMarkup()
    if (!svgContent) return
    downloadFile(svgContent, 'diagram.xml', 'application/xml')
  }

  const handleDownloadXaml = () => {
    const svgContent = getCurrentSvgMarkup()
    if (!svgContent) return
    downloadFile(svgContent, 'diagram.xaml', 'application/xml')
  }

  const handleDownloadPng = () => {
    const svgContent = getCurrentSvgMarkup()
    if (!svgContent) return
    const base = baseBoxRef.current || { w: 1200, h: 800 }
    const scale = 2
    const canvas = document.createElement('canvas')
    canvas.width = base.w * scale
    canvas.height = base.h * scale
    const ctx = canvas.getContext('2d')
    const img = new Image()
    const svgBlob = new Blob([svgContent], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(svgBlob)

    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
      canvas.toBlob((blob) => {
        if (blob) downloadFile(blob, 'diagram.png', 'image/png')
        URL.revokeObjectURL(url)
      }, 'image/png')
    }

    img.onerror = () => {
      console.error('PNG export failed')
      URL.revokeObjectURL(url)
    }

    img.src = url
  }

  const renderBody = () => {
    if (error) {
      return (
        <div className="mermaid-error-box">
          <div className="mermaid-error-title">Mermaid parse error</div>
          <div className="mermaid-error-message">{error}</div>
          <details className="mermaid-error-details">
            <summary>Show source</summary>
            <pre className="language-mermaid"><code>{chartCode}</code></pre>
          </details>
        </div>
      )
    }

    if (!svg) {
      return <div className="mermaid-wrapper">Rendering diagram...</div>
    }

    return (
      <>
        <div className="mermaid-toolbar">
          <button type="button" onClick={handleReset}>Reset</button>
          <span className="hint">Shift + Scroll to zoom, drag to pan</span>
        </div>
        <div
          className="mermaid-panzoom"
          key={chartCode}
          ref={containerRef}
          onWheelCapture={handleWheel}
          onMouseDown={handleMouseDown}
        >
          <div
            className="mermaid-panzoom-inner"
            dangerouslySetInnerHTML={{ __html: svg }}
          />
        </div>
      </>
    )
  }

  return (
    <div className="mermaid-wrapper">
      {adminMode && (
        <div className="mermaid-admin-bar">
          <div className="mermaid-admin-left">
            <button type="button" onClick={() => setShowCode(!showCode)}>
              {showCode ? 'Hide code' : 'Edit code'}
            </button>
            {showCode && draftCode !== chartCode && (
              <button type="button" onClick={handleApplyDraft}>Render update</button>
            )}
          </div>
          <span className="hint">Admin-only edit</span>
        </div>
      )}
      {adminMode && showCode && (
        <div className="mermaid-code-editor">
          <textarea
            className="mermaid-code-textarea"
            value={draftCode}
            onChange={(e) => setDraftCode(e.target.value)}
            spellCheck={false}
          />
        </div>
      )}
      {renderBody()}
      <div className="mermaid-actions">
        <button type="button" onClick={handleCopyCode}>Copy code</button>
        <button type="button" onClick={handleDownloadSvg} disabled={!svg}>Save SVG</button>
        <button type="button" onClick={handleDownloadPng} disabled={!svg}>Save PNG</button>
        <button type="button" onClick={handleDownloadXml} disabled={!svg}>Save XML</button>
        <button type="button" onClick={handleDownloadXaml} disabled={!svg}>Save XAML</button>
        <span className="mermaid-caption">with Mermaid</span>
      </div>
    </div>
  )
}

function CodeBlock({ language, codeText }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(codeText || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-lang">{language || 'text'}</span>
        <button className="code-copy-btn" onClick={handleCopy} title="Copy code">
          {copied ? <Check size={14} className="copied-icon" /> : <Copy size={14} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <div className="code-block-content">
        <SyntaxHighlighter
          language={language || 'text'}
          style={vscDarkPlus}
          showLineNumbers={true}
          customStyle={{ margin: 0, padding: 0, background: 'transparent', fontSize: '0.9rem' }}
          lineNumberStyle={{ minWidth: '2.5em', paddingRight: '1em', color: '#6e7681', textAlign: 'right', userSelect: 'none', opacity: 0.7 }}
          wrapLines={true}
          PreTag="div"
        >
          {codeText}
        </SyntaxHighlighter>
      </div>
    </div>
  )
}

export function MessageContent({ content, streaming = false, adminMode = false, onContentUpdate }) {
  // Функция для парсинга контента и выделения блока <think>
  const parseContent = (text) => {
    const thinkMatch = text.match(/<think>([\s\S]*?)(?:<\/think>|$)/)
    const thinkContent = thinkMatch ? thinkMatch[1] : null
    
    // Удаляем блок <think> из основного текста для отображения
    const mainContent = text.replace(/<think>[\s\S]*?(?:<\/think>|$)/, '').trim()
    
    return { thinkContent, mainContent }
  }

  const { thinkContent, mainContent } = parseContent(content)
  const [isThinkOpen, setIsThinkOpen] = useState(true)

  const markdownComponents = {
    code({ inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '')
      const language = match ? match[1].toLowerCase() : ''
      const codeText = String(children).trim()
      const looksMermaid =
        !inline &&
        (language === 'mermaid' || (!language && /^(graph|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|gitGraph)\b/i.test(codeText)))

      // Avoid rendering Mermaid while the message is streaming to prevent partial syntax errors and reflows
      if (looksMermaid && !streaming) {
        const handleCodeChange = (nextCode, prevCode) => {
          if (!onContentUpdate) return
          const previous = prevCode || codeText
          let newContent
          if ((previous && content.includes(previous)) || content.includes(codeText)) {
            const needle = content.includes(previous) ? previous : codeText
            newContent = content.replace(needle, nextCode)
          } else {
            newContent = `${content}\n\n\`\`\`mermaid\n${nextCode}\n\`\`\``
          }
          onContentUpdate(newContent)
        }

        return <MermaidBlock chart={codeText} adminMode={adminMode} onCodeChange={handleCodeChange} />
      }

      if (!inline) {
        return (
          <CodeBlock language={language} codeText={codeText} />
        )
      }

      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }
  }

  return (
    <div className="message-text-content">
      {thinkContent && (
        <div className="think-block">
          <div 
            className="think-header" 
            onClick={() => setIsThinkOpen(!isThinkOpen)}
          >
            {isThinkOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <span>Thinking Process</span>
          </div>
          {isThinkOpen && (
            <div className="think-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {thinkContent}
              </ReactMarkdown>
            </div>
          )}
        </div>
      )}
      
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
          {mainContent}
        </ReactMarkdown>
      </div>
    </div>
  )
}
