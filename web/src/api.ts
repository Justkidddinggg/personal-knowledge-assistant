import type { AskOut, DeleteOut, DocumentOut, UploadOut } from './types'

const BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://127.0.0.1:8000'

/** A failed call, with a message that is safe to show to the user. */
export class ApiError extends Error {
  status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const data: unknown = await response.json()
    if (typeof data === 'object' && data !== null && 'detail' in data) {
      const detail = (data as { detail: unknown }).detail
      if (typeof detail === 'string') return detail
      if (Array.isArray(detail) && detail.length > 0) {
        const first = detail[0] as { msg?: unknown }
        return typeof first?.msg === 'string'
          ? `Invalid request: ${first.msg}`
          : 'Invalid request'
      }
    }
  } catch {
    // the body was not JSON; fall through to the generic message
  }
  return `Request failed (HTTP ${response.status})`
}

async function request<T>(path: string, init: RequestInit, timeoutMs: number): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'TimeoutError') {
      throw new ApiError('The backend took too long to respond.')
    }
    throw new ApiError(`Cannot reach the backend at ${BASE_URL}. Is it running?`)
  }
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status)
  }
  return (await response.json()) as T
}

export const api = {
  listDocuments: () => request<DocumentOut[]>('/documents', {}, 10_000),

  uploadDocument: (file: File) => {
    const form = new FormData()
    form.append('file', file) // do NOT set Content-Type: the browser adds the boundary
    return request<UploadOut>('/documents', { method: 'POST', body: form }, 300_000)
  },

  deleteDocument: (docId: string) =>
    request<DeleteOut>(`/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' }, 30_000),

  ask: (question: string, docIds: string[] | null, sessionId: string | null) =>
    request<AskOut>(
      '/ask',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, doc_ids: docIds, session_id: sessionId }),
      },
      600_000,
    ),
}