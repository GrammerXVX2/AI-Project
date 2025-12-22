import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ChevronDown, ChevronRight } from 'lucide-react'
import './MessageContent.css'

export function MessageContent({ content }) {
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
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {thinkContent}
              </ReactMarkdown>
            </div>
          )}
        </div>
      )}
      
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {mainContent}
        </ReactMarkdown>
      </div>
    </div>
  )
}
