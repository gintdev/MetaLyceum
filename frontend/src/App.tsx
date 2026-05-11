import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import logo from './assets/img/logo.png'

type QueryModeValue = 0 | 1 | 2

type User = {
  id: number
  email: string
  display_name?: string | null
  avatar_url?: string | null
  created_at: string
}

type AuthResponse = {
  access_token: string
  token_type: 'bearer'
  user: User
}

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (options: {
            client_id: string
            callback: (response: { credential: string }) => void
            auto_select?: boolean
            cancel_on_tap_outside?: boolean
          }) => void
          renderButton: (
            parent: HTMLElement,
            options: {
              type?: 'standard' | 'icon'
              theme?: 'outline' | 'filled_blue' | 'filled_black'
              size?: 'large' | 'medium' | 'small'
              shape?: 'rectangular' | 'pill' | 'circle' | 'square'
              text?: 'signin_with' | 'signup_with' | 'continue_with' | 'signin'
              width?: string
            }
          ) => void
        }
      }
    }
  }
}

type Chat = {
  id: number
  title: string
  created_at: string
  updated_at: string
  last_message_preview?: string | null
}

type Message = {
  id: number
  chat_id: number
  role: 'user' | 'assistant' | 'system'
  content: string
  sources: Array<Record<string, unknown>>
  query_type?: QueryModeValue
  created_at: string
}

type ChatMessagesResponse = {
  items: Message[]
}

type AskResponse = {
  chat_id: number
  query: string
  answer: string
  sources: Array<Record<string, unknown>>
  pdf_files?: Array<{ filename: string; article_id?: number; display_name?: string; ref_index?: number }>
  chunks_used: number
  status: string
}

type QueryMode = {
  label: 'Вопрос' | 'Эссе' | 'Литература'
  value: QueryModeValue
}

const QUERY_MODES: QueryMode[] = [
  { label: 'Вопрос', value: 0 },
  { label: 'Эссе', value: 1 },
  { label: 'Литература', value: 2 }
]

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? ''
const TOKEN_KEY = 'metalyceum_jwt'

function extractAttachmentsFromSources(sources: Array<Record<string, unknown>>): Array<{ filename: string; display_name?: string }> {
  const seen = new Set<string>()
  const files: Array<{ filename: string; display_name?: string }> = []

  for (const source of sources) {
    const filename = typeof source.filename === 'string' ? source.filename : null
    const displayName = typeof source.display_name === 'string' ? source.display_name : null
    if (!filename || seen.has(filename)) continue
    seen.add(filename)
    files.push({ filename, display_name: displayName ?? undefined })
  }

  return files
}

function formatSidebarUserName(user: User | null): string {
  if (!user) return ''

  const displayName = user.display_name?.trim()
  if (!displayName) {
    return user.email
  }

  const parts = displayName.split(/\s+/).filter((part) => part.length > 0)
  if (parts.length < 2) {
    return parts[0] ?? user.email
  }

  return `${parts[0]} ${parts[1][0]}`
}

export default function App() {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY))
  const [user, setUser] = useState<User | null>(null)
  const [isAuthLoading, setIsAuthLoading] = useState(false)
  const [authError, setAuthError] = useState<string | null>(null)
  const googleButtonRef = useRef<HTMLDivElement | null>(null)

  const [chats, setChats] = useState<Chat[]>([])
  const [activeChatId, setActiveChatId] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [isHistoryLoading, setIsHistoryLoading] = useState(false)

  const [input, setInput] = useState('')
  const [keywordsInput, setKeywordsInput] = useState('')
  const [keywordSuggestions, setKeywordSuggestions] = useState<string[]>([])
  const [showKeywordSuggestions, setShowKeywordSuggestions] = useState(false)
  const [sourceOptions, setSourceOptions] = useState<string[]>([])
  const [selectedSources, setSelectedSources] = useState<string[]>([])
  const [isSourceDropdownOpen, setIsSourceDropdownOpen] = useState(false)
  const [publicationDateFrom, setPublicationDateFrom] = useState('')
  const [publicationDateTo, setPublicationDateTo] = useState('')
  const [queryType, setQueryType] = useState<QueryModeValue>(0)
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const keywordQuery = useMemo(() => {
    const parts = keywordsInput.split(',')
    const lastPart = parts[parts.length - 1] ?? ''
    return lastPart.trim()
  }, [keywordsInput])

  const canSend = input.trim().length > 0 && !isSending

  const sourceButtonLabel = useMemo(() => {
    if (selectedSources.length === 0) return 'Источник'
    if (selectedSources.length === 1) return selectedSources[0]
    return `Источников: ${selectedSources.length}`
  }, [selectedSources])

  const currentChat = chats.find((chat) => chat.id === activeChatId) ?? null
  const sidebarUserLabel = formatSidebarUserName(user)

  async function requestJson<T>(path: string, options: RequestInit = {}, needsAuth = false): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> | undefined)
    }

    if (needsAuth && token) {
      headers.Authorization = `Bearer ${token}`
    }

    const response = await fetch(`${API_URL}${path}`, {
      ...options,
      headers
    })

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}))
      const detail = typeof errBody?.detail === 'string' ? errBody.detail : 'Ошибка запроса'

      if (response.status === 401 && needsAuth) {
        localStorage.removeItem(TOKEN_KEY)
        setToken(null)
        setUser(null)
      }

      throw new Error(detail)
    }

    if (response.status === 204) {
      return undefined as T
    }

    return (await response.json()) as T
  }

  async function loadChats() {
    const nextChats = await requestJson<Chat[]>('/chats/', {}, true)
    setChats(nextChats)
    if (nextChats.length === 0) {
      setActiveChatId(null)
      setMessages([])
      return
    }
    if (!activeChatId || !nextChats.some((chat) => chat.id === activeChatId)) {
      setActiveChatId(nextChats[0].id)
    }
  }

  async function loadMessages(chatId: number) {
    setIsHistoryLoading(true)
    try {
      const response = await requestJson<ChatMessagesResponse>(`/chats/${chatId}/messages`, {}, true)
      setMessages(response.items)
    } finally {
      setIsHistoryLoading(false)
    }
  }

  async function createChat(title?: string): Promise<Chat> {
    const chat = await requestJson<Chat>(
      '/chats/',
      {
        method: 'POST',
        body: JSON.stringify({ title })
      },
      true
    )
    setChats((prev) => [chat, ...prev])
    setActiveChatId(chat.id)
    setMessages([])
    return chat
  }

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

  useEffect(() => {
    if (!token) {
      setUser(null)
      setChats([])
      setActiveChatId(null)
      setMessages([])
      return
    }

    const bootstrap = async () => {
      try {
        const me = await requestJson<User>('/auth/me', {}, true)
        setUser(me)
        await loadChats()
      } catch (bootstrapError) {
        const message = bootstrapError instanceof Error ? bootstrapError.message : 'Не удалось загрузить профиль'
        setAuthError(message)
      }
    }

    void bootstrap()
  }, [token])

  useEffect(() => {
    if (!activeChatId) return
    void loadMessages(activeChatId)
  }, [activeChatId])

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

  const toggleSourceSelection = (sourceValue: string) => {
    setSelectedSources((prev) =>
      prev.includes(sourceValue) ? prev.filter((item) => item !== sourceValue) : [...prev, sourceValue]
    )
  }

  const handleGoogleAuth = async (credential: string) => {
    setAuthError(null)
    setIsAuthLoading(true)
    try {
      const result = await requestJson<AuthResponse>('/auth/google', {
        method: 'POST',
        body: JSON.stringify({ credential })
      })

      localStorage.setItem(TOKEN_KEY, result.access_token)
      setToken(result.access_token)
      setUser(result.user)
    } catch (submitError) {
      setAuthError(submitError instanceof Error ? submitError.message : 'Ошибка авторизации')
    } finally {
      setIsAuthLoading(false)
    }
  }

  useEffect(() => {
    if (token || !googleButtonRef.current) return

    const container = googleButtonRef.current
    container.innerHTML = ''

    if (!GOOGLE_CLIENT_ID) {
      setAuthError('Не задан VITE_GOOGLE_CLIENT_ID для входа через Google')
      return
    }

    if (!window.google?.accounts?.id) {
      setAuthError('Google Identity Services не загружен. Проверьте подключение скрипта.')
      return
    }

    window.google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: (response) => {
        if (!response.credential) {
          setAuthError('Google не вернул credential token')
          return
        }
        void handleGoogleAuth(response.credential)
      },
      cancel_on_tap_outside: true
    })

    window.google.accounts.id.renderButton(container, {
      type: 'standard',
      theme: 'outline',
      size: 'large',
      text: 'signin_with',
      shape: 'pill',
      width: '360'
    })
  }, [token])

  const handleLogout = () => {
    localStorage.removeItem(TOKEN_KEY)
    setToken(null)
    setUser(null)
    setChats([])
    setActiveChatId(null)
    setMessages([])
  }

  const handleCreateChat = async () => {
    try {
      await createChat('Новый чат')
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Не удалось создать чат')
    }
  }

  const handleDeleteChat = async (chatId: number) => {
    try {
      await requestJson<void>(`/chats/${chatId}`, { method: 'DELETE' }, true)
      const nextChats = chats.filter((chat) => chat.id !== chatId)
      setChats(nextChats)
      if (activeChatId === chatId) {
        setActiveChatId(nextChats[0]?.id ?? null)
        if (nextChats.length === 0) {
          setMessages([])
        }
      }
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Не удалось удалить чат')
    }
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const text = input.trim()
    if (!text) return

    setError(null)
    setIsSending(true)
    let chatIdForRequest: number | null = null
    let optimisticMessageId: number | null = null

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

      if (keywords.length > 0) payload.keywords = keywords
      if (selectedSources.length > 0) payload.sources = selectedSources

      const yearFrom = Number.parseInt(publicationDateFrom, 10)
      if (!Number.isNaN(yearFrom)) payload.publication_date_from = yearFrom

      const yearTo = Number.parseInt(publicationDateTo, 10)
      if (!Number.isNaN(yearTo)) payload.publication_date_to = yearTo

      let chatId = activeChatId
      if (!chatId) {
        const created = await createChat(text.slice(0, 60))
        chatId = created.id
      }
      chatIdForRequest = chatId

      optimisticMessageId = Date.now()
      const optimisticUserMessage: Message = {
        id: optimisticMessageId,
        chat_id: chatId,
        role: 'user',
        content: text,
        sources: [],
        query_type: queryType,
        created_at: new Date().toISOString()
      }

      setMessages((prev) => [...prev, optimisticUserMessage])

      setInput('')

      const result = await requestJson<AskResponse>(
        `/chats/${chatId}/ask`,
        {
          method: 'POST',
          body: JSON.stringify(payload)
        },
        true
      )

      await loadMessages(result.chat_id)
      await loadChats()
      setActiveChatId(result.chat_id)
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : 'Не удалось отправить сообщение'
      setError(message)
      if (optimisticMessageId !== null) {
        setMessages((prev) => prev.filter((item) => item.id !== optimisticMessageId))
      }
      if (chatIdForRequest !== null) {
        await loadMessages(chatIdForRequest).catch(() => undefined)
      }
    } finally {
      setIsSending(false)
    }
  }

  if (!token || !user) {
    return (
      <main className="authPage">
        <div className="authGlow authGlowLeft" />
        <div className="authGlow authGlowRight" />
        <section className="authCard">
          <img src={logo} alt="MetaLyceum" className="authLogo" />
          <h1>MetaLyceum</h1>
          <p className="authSubtitle">Войдите, чтобы продолжить</p>

          <div className="authGoogleWrap">
            {isAuthLoading && <div className="authHint">Проверяем Google токен...</div>}
            <div ref={googleButtonRef} className="googleButtonContainer" />
          </div>

          {authError && <div className="authError">{authError}</div>}
        </section>
      </main>
    )
  }

  return (
    <main className="appLayout">
      <aside className="sidebar">
        <div className="sidebarTop">
          <div className="brand">
            <img src={logo} alt="MetaLyceum" className="brandLogo" />
            <div className="brandName">MetaLyceum</div>
          </div>

          <button type="button" className="newChatButton" onClick={handleCreateChat}>
            + Новый чат
          </button>
        </div>

        <div className="chatList" role="list">
          {chats.map((chat) => (
            <div key={chat.id} className={chat.id === activeChatId ? 'chatListItem active' : 'chatListItem'} role="listitem">
              <button type="button" className="chatSelect" onClick={() => setActiveChatId(chat.id)}>
                <div className="chatTitle">{chat.title}</div>
                {chat.last_message_preview && <div className="chatPreview">{chat.last_message_preview}</div>}
              </button>
              <button
                type="button"
                className="chatDelete"
                onClick={() => void handleDeleteChat(chat.id)}
                aria-label="Удалить чат"
                title="Удалить чат"
              >
                ✕
              </button>
            </div>
          ))}
          {chats.length === 0 && <div className="chatEmpty">Создайте первый чат</div>}
        </div>

        <div className="sidebarFooter">
          <div className="sidebarUser" title={user.display_name || user.email}>
            {sidebarUserLabel}
          </div>
          <button type="button" className="logoutButton" onClick={handleLogout}>
            Выйти
          </button>
        </div>
      </aside>

      <section className="workspace">
        <header className="workspaceHeader">
          <h1>{currentChat?.title || 'Новый чат'}</h1>
        </header>

        <section className="modeSwitch" aria-label="Выбор режима ответа">
          {QUERY_MODES.map((mode) => (
            <button
              key={mode.value}
              type="button"
              className={mode.value === queryType ? 'modeButton modeButtonActive' : 'modeButton'}
              onClick={() => setQueryType(mode.value)}
              disabled={isSending}
            >
              {mode.label}
            </button>
          ))}
        </section>

        <section className="chatPane">
          {isHistoryLoading && <p className="placeholder">Загрузка истории...</p>}
          {!isHistoryLoading && messages.length === 0 && <p className="placeholder">cogito, ergo sum</p>}

          {!isHistoryLoading &&
            messages.map((message) => {
              const fallbackAttachments = extractAttachmentsFromSources(message.sources)
              return (
                <article
                  key={message.id}
                  className={message.role === 'user' ? 'message messageUser' : 'message messageAssistant'}
                >
                  <div className="messageRole">{message.role === 'user' ? 'Вы' : 'Диоген'}</div>
                  <div className="messageText">{message.content}</div>

                  {message.role === 'assistant' && fallbackAttachments.length > 0 && (
                    <div className="messageAttachments">
                      <div className="messageAttachmentsTitle">PDF источники:</div>
                      {fallbackAttachments.map((file) => (
                        <a
                          key={`${message.id}-${file.filename}`}
                          href={`${API_URL}/articles/files/source-pdf?filename=${encodeURIComponent(file.filename)}`}
                          target="_blank"
                          rel="noreferrer"
                          className="messageAttachmentLink"
                        >
                          {file.display_name ?? file.filename}
                        </a>
                      ))}
                    </div>
                  )}
                </article>
              )
            })}
        </section>

        <form className="composer" onSubmit={handleSubmit}>
          <div className="filterRow">
            <input
              className="filterInput"
              placeholder="Ключевые слова"
              value={keywordsInput}
              onChange={(event) => setKeywordsInput(event.target.value)}
              onFocus={() => setShowKeywordSuggestions(true)}
              onBlur={() => window.setTimeout(() => setShowKeywordSuggestions(false), 120)}
              disabled={isSending}
            />

            {showKeywordSuggestions && keywordSuggestions.length > 0 && (
              <ul className="keywordSuggestions" role="listbox" aria-label="Подсказки ключевых слов">
                {keywordSuggestions.map((item) => (
                  <li key={item}>
                    <button type="button" className="keywordSuggestionItem" onMouseDown={() => applyKeywordSuggestion(item)}>
                      {item}
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <div className="sourceDropdown">
              <button
                type="button"
                className="sourceDropdownButton"
                onClick={() => setIsSourceDropdownOpen((prev) => !prev)}
                disabled={isSending}
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

            <input
              className="yearInput"
              type="number"
              placeholder="Год от"
              value={publicationDateFrom}
              onChange={(event) => setPublicationDateFrom(event.target.value)}
              disabled={isSending}
            />
            <input
              className="yearInput"
              type="number"
              placeholder="Год до"
              value={publicationDateTo}
              onChange={(event) => setPublicationDateTo(event.target.value)}
              disabled={isSending}
            />
          </div>

          <div className="composerMain">
            <textarea
              className="composerInput"
              placeholder="Напишите вопрос..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              rows={3}
            />
            <button className="sendButton" type="submit" disabled={!canSend}>
              {isSending ? '...' : 'Отправить'}
            </button>
          </div>
        </form>

        {error && <p className="errorText">{error}</p>}
      </section>
    </main>
  )
}
