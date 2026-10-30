import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api'
import type { DocumentOut } from '../types'

const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt', '.md']

type Notice = { kind: 'success' | 'info' | 'error'; text: string }

const NOTICE_STYLES: Record<Notice['kind'], string> = {
  success: 'bg-green-50 text-green-800',
  info: 'bg-blue-50 text-blue-800',
  error: 'bg-red-50 text-red-800',
}

// A convenience for the user only; the backend validates the real file type.
function hasAllowedExtension(name: string): boolean {
  const lower = name.toLowerCase()
  return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext))
}

function describe(error: unknown): string {
  return error instanceof ApiError ? error.message : 'Something went wrong.'
}

export default function DocumentPanel() {
  const [docs, setDocs] = useState<DocumentOut[] | null>(null) // null = still loading
  const [listError, setListError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [notices, setNotices] = useState<Notice[]>([])
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let cancelled = false
    api
      .listDocuments()
      .then((result) => {
        if (cancelled) return
        setDocs(result)
        setListError(null)
      })
      .catch((error: unknown) => {
        if (!cancelled) setListError(describe(error))
      })
    return () => {
      cancelled = true
    }
  }, [refreshKey])

  const refresh = () => setRefreshKey((key) => key + 1)

  async function uploadFiles(fileList: FileList | null) {
    const files = Array.from(fileList ?? [])
    if (files.length === 0) return
    setUploading(true)
    const results: Notice[] = []
    for (const file of files) {
      if (!hasAllowedExtension(file.name)) {
        results.push({
          kind: 'error',
          text: `${file.name}: unsupported file type (use PDF, DOCX, TXT or MD).`,
        })
        continue
      }
      try {
        const out = await api.uploadDocument(file)
        results.push(
          out.status === 'duplicate'
            ? { kind: 'info', text: `${file.name}: already indexed (identical content).` }
            : { kind: 'success', text: `${file.name}: indexed (${out.num_chunks} chunks).` },
        )
      } catch (error) {
        results.push({ kind: 'error', text: `${file.name}: ${describe(error)}` })
      }
    }
    setNotices(results)
    setUploading(false)
    refresh()
    if (inputRef.current) inputRef.current.value = '' // allow choosing the same file again
  }

  async function remove(doc: DocumentOut) {
    if (!window.confirm(`Delete "${doc.filename}" and its index?`)) return
    try {
      await api.deleteDocument(doc.doc_id)
      setNotices([{ kind: 'success', text: `${doc.filename}: deleted.` }])
    } catch (error) {
      setNotices([{ kind: 'error', text: `${doc.filename}: ${describe(error)}` }])
    }
    refresh()
  }

  return (
    <div className="space-y-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
        Documents
      </h2>

      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
        }}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          void uploadFiles(e.dataTransfer.files)
        }}
        className={`cursor-pointer rounded-lg border-2 border-dashed px-4 py-6 text-center text-sm transition-colors ${
          dragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-white hover:bg-slate-50'
        }`}
      >
        {uploading ? 'Indexing... (the first upload can take a while)' : 'Drop files here or click to choose'}
        <div className="mt-1 text-xs text-slate-500">PDF, DOCX, TXT, MD</div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.md"
          className="hidden"
          onChange={(e) => void uploadFiles(e.target.files)}
        />
      </div>

      {notices.length > 0 && (
        <ul className="space-y-1">
          {notices.map((notice, index) => (
            <li key={index} className={`rounded px-3 py-2 text-xs ${NOTICE_STYLES[notice.kind]}`}>
              {notice.text}
            </li>
          ))}
        </ul>
      )}

      {listError && (
        <div className="rounded bg-red-50 px-3 py-2 text-xs text-red-800">
          {listError}{' '}
          <button className="underline" onClick={refresh}>
            Retry
          </button>
        </div>
      )}

      {docs === null && !listError && <p className="text-sm text-slate-500">Loading...</p>}
      {docs !== null && docs.length === 0 && (
        <p className="text-sm text-slate-500">No documents yet. Upload one above.</p>
      )}

      {docs !== null && docs.length > 0 && (
        <ul className="divide-y rounded-lg border bg-white">
          {docs.map((doc) => (
            <li key={doc.doc_id} className="flex items-center gap-2 px-3 py-2">
              <div className="min-w-0 flex-1">
                {/* React escapes text, so file names can never inject markup */}
                <div className="truncate text-sm" title={doc.filename}>
                  {doc.filename}
                </div>
                <div className="text-xs text-slate-500">{doc.num_chunks} chunks</div>
              </div>
              <button
                onClick={() => void remove(doc)}
                className="rounded px-2 py-1 text-xs text-red-700 hover:bg-red-50"
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}