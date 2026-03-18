import { FormEvent, useEffect, useMemo, useState } from 'react'
import logo from './assets/img/logo.png'

type Message = {
  id: number
  role: 'user' | 'assistant'
  text: string
  attachments?: Array<{ filename: string; article_id?: number; display_name?: string; ref_index?: number }>
}

type AskResponse = {
  query: string
  answer: string
  sources: Array<Record<string, unknown>>
  pdf_files?: Array<{ filename: string; article_id?: number; display_name?: string; ref_index?: number }>
  chunks_used: number
  status: string
}

type QueryMode = {
  label: 'Вопрос' | 'Эссе' | 'Литература'
  value: 0 | 1 | 2
}

const QUERY_MODES: QueryMode[] = [
  { label: 'Вопрос', value: 0 },
  { label: 'Эссе', value: 1 },
  { label: 'Литература', value: 2 }
]

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [keywordsInput, setKeywordsInput] = useState('')
  const [keywordSuggestions, setKeywordSuggestions] = useState<string[]>([])
  const [showKeywordSuggestions, setShowKeywordSuggestions] = useState(false)
  const [sourceOptions, setSourceOptions] = useState<string[]>([])
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [isSourceDropdownOpen, setIsSourceDropdownOpen] = useState(false)
  const [publicationDateFrom, setPublicationDateFrom] = useState('')
  const [publicationDateTo, setPublicationDateTo] = useState('')
  const [queryType, setQueryType] = useState<QueryMode['value']>(0)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const keywordQuery = useMemo(() => {
    const parts = keywordsInput.split(',')
    const lastPart = parts[parts.length - 1] ?? ''
    return lastPart.trim()
  }, [keywordsInput])

  useEffect(() => {
    const controller = new AbortController()
    const timeoutId = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({ limit: '10' })
        if (keywordQuery.length > 0) {
          params.set('query', keywordQuery)
        }

        const response = await fetch(`${API_URL}/articles/keywords?${params.toString()}`, {
          signal: controller.signal
        })

        if (!response.ok) {
          setKeywordSuggestions([])
          return
        }

        const data = (await response.json()) as string[]
        setKeywordSuggestions(data)
      } catch {
        setKeywordSuggestions([])
      }
    }, 250)

    return () => {
      window.clearTimeout(timeoutId)
      controller.abort()
    }
  }, [keywordQuery])

  useEffect(() => {
    const controller = new AbortController()

    const loadSources = async () => {
      try {
        const response = await fetch(`${API_URL}/articles/sources?limit=500`, {
          signal: controller.signal
        })

        if (!response.ok) {
          setSourceOptions([])
          return
        }

        const data = (await response.json()) as string[]
        setSourceOptions(data)
      } catch {
        setSourceOptions([])
      }
    }

    void loadSources()

    return () => controller.abort()
  }, [])

  const applyKeywordSuggestion = (suggestion: string) => {
    const parts = keywordsInput
      .split(',')
      .slice(0, -1)
      .map((part) => part.trim())
      .filter((part) => part.length > 0)

    const nextValue = [...parts, suggestion].join(', ')
    setKeywordsInput(nextValue.length > 0 ? `${nextValue}, ` : '')
    setShowKeywordSuggestions(false)
  }

  const canSend = useMemo(() => input.trim().length > 0 && !isLoading, [input, isLoading])

  const sourceButtonLabel = useMemo(() => {
    if (selectedSources.length === 0) return 'источник'
    if (selectedSources.length === 1) return selectedSources[0]
    return `Выбрано источников: ${selectedSources.length}`
  }, [selectedSources])

  const toggleSourceSelection = (sourceValue: string) => {
    setSelectedSources((prev) =>
      prev.includes(sourceValue)
        ? prev.filter((item) => item !== sourceValue)
        : [...prev, sourceValue]
    )
  }

  const handleLogoClick = () => {
    window.location.reload()
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const text = input.trim()
    if (!text) return

    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      text
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setError(null)
    setIsLoading(true)

    try {
      const keywords = keywordsInput
        .split(',')
        .map((item) => item.trim())
        .filter((item) => item.length > 0)

      const payload: Record<string, unknown> = {
        query: text,
        limit: 5,
        query_type: queryType
      }

      if (keywords.length > 0) {
        payload.keywords = keywords
      }

      if (selectedSources.length > 0) {
        payload.sources = selectedSources
      }

      const yearFrom = Number.parseInt(publicationDateFrom, 10)
      if (!Number.isNaN(yearFrom)) {
        payload.publication_date_from = yearFrom
      }

      const yearTo = Number.parseInt(publicationDateTo, 10)
      if (!Number.isNaN(yearTo)) {
        payload.publication_date_to = yearTo
      }

      const response = await fetch(`${API_URL}/articles/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        const detail = typeof errBody?.detail === 'string' ? errBody.detail : 'Ошибка запроса к API'
        throw new Error(detail)
      }

      const data = (await response.json()) as AskResponse

      const botMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        text: data.answer || 'Пустой ответ от сервера',
        attachments: data.pdf_files ?? []
      }

      setMessages((prev) => [...prev, botMessage])
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : 'Не удалось отправить сообщение'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="container">
      <header className="brandHeader">
        <button
          type="button"
          className="logoButton"
          onClick={handleLogoClick}
          aria-label="Обновить страницу"
          title="Обновить страницу"
        >
          <img src={logo} alt="MetaLyceum" className="logoImage" />
        </button>

        <div className="brandText">
          <h1 className="title">MetaLyceum</h1>
          <p className="subtitle">AI philosophy teacher</p>
        </div>
      </header>

      <section className="modeSwitch" aria-label="Выбор режима ответа">
        {QUERY_MODES.map((mode) => (
          <button
            key={mode.value}
            type="button"
            className={mode.value === queryType ? 'modeButton modeButtonActive' : 'modeButton'}
            onClick={() => setQueryType(mode.value)}
            disabled={isLoading}
          >
            {mode.label}
          </button>
        ))}
      </section>

      <section className="chatBox">
        {messages.length === 0 && <p className="placeholder">Cogito, ergo sum</p>}
        {messages.map((message) => (
          <article
            key={message.id}
            className={message.role === 'user' ? 'message messageUser' : 'message messageAssistant'}
          >
            <div className="messageRole">{message.role === 'user' ? 'Вы' : 'Диоген'}</div>
            <div>{message.text}</div>
            {message.role === 'assistant' && (message.attachments?.length ?? 0) > 0 && (
              <div className="messageAttachments">
                <div className="messageAttachmentsTitle">PDF источники:</div>
                {message.attachments?.map((file, index) => (
                  <a
                    key={`${file.filename}-${index}`}
                    href={`${API_URL}/articles/files/source-pdf?filename=${encodeURIComponent(file.filename)}`}
                    target="_blank"
                    rel="noreferrer"
                    className="messageAttachmentLink"
                  >
                    [{file.ref_index ?? index + 1}] {file.display_name ?? file.filename}
                  </a>
                ))}
              </div>
            )}
          </article>
        ))}
      </section>

      <form className="form" onSubmit={handleSubmit}>
        <textarea
          className="input"
          placeholder="Спросите MetaLyceum..."
          value={input}
          onChange={(event) => setInput(event.target.value)}
          rows={3}
        />
        <section className="filterPanel" aria-label="Фильтры поиска">
          <div className="filterTitle">Фильтры</div>
          <div className="filterGrid">
            <input
              className="filterInput"
              placeholder="ключевые слова"
              value={keywordsInput}
              onChange={(event) => setKeywordsInput(event.target.value)}
              onFocus={() => setShowKeywordSuggestions(true)}
              onBlur={() => window.setTimeout(() => setShowKeywordSuggestions(false), 120)}
              disabled={isLoading}
            />
            {showKeywordSuggestions && keywordSuggestions.length > 0 && (
              <ul className="keywordSuggestions" role="listbox" aria-label="Подсказки ключевых слов">
                {keywordSuggestions.map((item) => (
                  <li key={item}>
                    <button
                      type="button"
                      className="keywordSuggestionItem"
                      onMouseDown={() => applyKeywordSuggestion(item)}
                    >
                      {item}
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <div className="sourceDropdown">
              <button
                type="button"
                className="filterInput sourceDropdownButton"
                onClick={() => setIsSourceDropdownOpen((prev) => !prev)}
                disabled={isLoading}
              >
                {sourceButtonLabel}
              </button>

              {isSourceDropdownOpen && (
                <div className="sourceDropdownMenu" role="listbox" aria-label="Список источников">
                  {sourceOptions.length === 0 && <div className="sourceEmpty">Нет доступных источников</div>}
                  {sourceOptions.map((item) => (
                    <label key={item} className="sourceOption">
                      <input
                        type="checkbox"
                        checked={selectedSources.includes(item)}
                        onChange={() => toggleSourceSelection(item)}
                      />
                      <span>{item}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            <div className="publicationYearSection">
              <div className="publicationYearTitle">Год публикации</div>
              <div className="publicationYearInputs">
                <input
                  className="filterInput"
                  type="number"
                  placeholder="С"
                  value={publicationDateFrom}
                  onChange={(event) => setPublicationDateFrom(event.target.value)}
                  disabled={isLoading}
                />
                <input
                  className="filterInput"
                  type="number"
                  placeholder="По"
                  value={publicationDateTo}
                  onChange={(event) => setPublicationDateTo(event.target.value)}
                  disabled={isLoading}
                />
              </div>
            </div>
          </div>
        </section>
        <button className="button" type="submit" disabled={!canSend}>
          {isLoading ? 'Отправка...' : 'Отправить'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}
    </main>
  )
}
