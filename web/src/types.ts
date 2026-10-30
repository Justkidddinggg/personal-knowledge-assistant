// These mirror the Pydantic schemas in app/schemas/api.py.

export interface DocumentOut {
  doc_id: string
  filename: string
  num_chunks: number
}

export interface UploadOut {
  doc_id: string
  filename: string
  status: 'indexed' | 'duplicate'
  num_chunks: number
}

export interface DeleteOut {
  doc_id: string
  deleted_chunks: number
}

export interface SourceOut {
  label: string // "S1", "S2", ... matches the [S#] citations in the answer
  doc_id: string
  chunk_id: string
  filename: string
  page: number | null
  score: number
  text: string
  cited: boolean
}

export interface AskOut {
  answer: string
  session_id: string
  search_query: string
  grounded: boolean
  sources: SourceOut[]
  invalid_citations: number[]
  retrieval_ms: number
  generation_ms: number
}