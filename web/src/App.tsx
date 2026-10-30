import DocumentPanel from './components/DocumentPanel'

export default function App() {
  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-900">
      <aside className="w-80 shrink-0 border-r bg-slate-100 p-4">
        <h1 className="mb-6 text-lg font-semibold">Personal Knowledge Assistant</h1>
        <DocumentPanel />
      </aside>
      <main className="flex flex-1 items-center justify-center p-6 text-slate-500">
        The chat arrives in the next step.
      </main>
    </div>
  )
}