import { Fragment, useEffect, useMemo, useState } from 'react'
import Editor from '@monaco-editor/react'
import { Activity, BookOpen, Braces, Check, ChevronDown, CircleHelp, Clock3, Code2, FlaskConical, GitBranch, Lightbulb, LoaderCircle, Play, RotateCcw, Send, Terminal, Waves, X } from 'lucide-react'

type Example = { input: string[]; output: (string | null)[] }
type Challenge = {
  id: string; title: string; difficulty: string; category: string; description: string; behavior: string; metric: string
  model: { name: string; layers: number; heads: number; d_model: number }
  examples: Example[]; learning_objectives: string[]; tools: string[]; hints: string[]; starter_code?: string; task_goal?: string; example_note?: string
  grading_contract: Array<{ field: string; type: string; graded: boolean; checks: string }>
  tool_reference: Array<{ name: string; type: string; description: string }>
  explanation?: { canonical_solution?: string; caution?: string }
}
type ChallengeSummary = Pick<Challenge, 'id' | 'title' | 'difficulty' | 'category'>
type HistoryRun = { run_id: string; challenge_id: string; status: string; created_at: string; result?: RunResult | null }
type RunResult = {
  run_id: string; status: 'complete' | 'error' | 'unavailable'; stdout?: string; stderr?: string; error?: string
  traceback?: string; events?: Array<Record<string, any>>; baseline?: { accuracy: number; correct: number; total: number }
  submission?: Record<string, any>; traces?: {
    attention?: { layer: number; head: number; tokens: string[]; weights: number[][] }
    value_path?: Record<string, any>; patching?: Record<string, any>
  }
}
type Grade = { submission_id: string; score: number; dimensions: Record<string, number>; checks: Record<string, any>; feedback: string[]; explanation?: Challenge['explanation'] }

const STARTER = `def solve(model, tokenizer, dataset, tools):
    # Try one PyTorch-backed model call and inspect candidate heads.
    tokens = dataset[0]["tokens"]
    logits = model(tokens)
    print("Logits shape:", tuple(logits.shape))

    # Ablate heads across public examples; the helper records these tests.
    scores = []
    for layer in range(model.n_layers):
        for head in range(model.n_heads):
            drops = [tools.ablate_head(row["tokens"], layer, head)["drop"] for row in dataset]
            scores.append((sum(drops) / len(drops), layer, head))
    _, layer, head = max(scores)
    print("Top heads:", sorted(scores, reverse=True)[:5])
    return {
        "components": [{"type": "attention_head", "layer": layer, "head": head}]
    }
`

function getSessionId() {
  const saved = sessionStorage.getItem('circuit-session')
  if (saved) return saved
  const id = crypto.randomUUID()
  sessionStorage.setItem('circuit-session', id)
  return id
}

function App() {
  const [challenge, setChallenge] = useState<Challenge | null>(null)
  const [challenges, setChallenges] = useState<ChallengeSummary[]>([])
  const [completedChallengeIds, setCompletedChallengeIds] = useState<Set<string>>(new Set())
  const [selectedChallengeId, setSelectedChallengeId] = useState('previous-token-head')
  const [apiState, setApiState] = useState<'loading' | 'ready' | 'offline'>('loading')
  const [code, setCode] = useState(STARTER)
  const [run, setRun] = useState<RunResult | null>(null)
  const [grade, setGrade] = useState<Grade | null>(null)
  const [busy, setBusy] = useState<'run' | 'submit' | null>(null)
  const [tab, setTab] = useState<'output' | 'attention' | 'value_path' | 'patching' | 'submission'>('output')
  const [hintIndex, setHintIndex] = useState(-1)
  const [toast, setToast] = useState('')
  const [historyOpen, setHistoryOpen] = useState(false)
  const [runHistory, setRunHistory] = useState<HistoryRun[]>([])
  const sessionId = useMemo(getSessionId, [])

  useEffect(() => {
    fetch('/api/challenges')
      .then((response) => { if (!response.ok) throw new Error('API unavailable'); return response.json() })
      .then((data: ChallengeSummary[]) => { setChallenges(data); if (data.length && !data.some((item) => item.id === selectedChallengeId)) setSelectedChallengeId(data[0].id) })
      .catch(() => setApiState('offline'))
  }, [])

  useEffect(() => {
    let active = true
    fetch(`/api/challenges/${selectedChallengeId}`)
      .then((response) => { if (!response.ok) throw new Error('API unavailable'); return response.json() })
      .then(async (data: Challenge) => {
        const draftResponse = await fetch(`/api/sessions/${sessionId}/drafts/${selectedChallengeId}`).catch(() => null)
        const draft = draftResponse?.ok ? await draftResponse.json() : null
        if (active) { setChallenge(data); setCode(draft?.source || data.starter_code || STARTER); setApiState('ready') }
      })
      .catch(() => { if (active) setApiState('offline') })
    return () => { active = false }
  }, [selectedChallengeId, sessionId])

  useEffect(() => {
    if (!challenge || challenge.id !== selectedChallengeId) return
    const timer = window.setTimeout(() => {
      fetch(`/api/sessions/${sessionId}/drafts/${selectedChallengeId}`, {
        method: 'PUT', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, source: code }),
      }).catch(() => undefined)
    }, 500)
    return () => window.clearTimeout(timer)
  }, [challenge, code, selectedChallengeId, sessionId])

  useEffect(() => {
    fetch(`/api/sessions/${sessionId}/submissions`).then((response) => response.ok ? response.json() : []).then((items: Array<{ challenge_id: string; score: number }>) => {
      setCompletedChallengeIds(new Set(items.filter((item) => item.score >= 80).map((item) => item.challenge_id)))
    }).catch(() => undefined)
  }, [sessionId])

  const notify = (message: string) => { setToast(message); window.setTimeout(() => setToast(''), 2600) }

  async function runCode() {
    setBusy('run'); setGrade(null); setTab('output')
    try {
      const response = await fetch('/api/runs', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ challenge_id: selectedChallengeId, session_id: sessionId, source: code }) })
      const data: RunResult = await response.json()
      setRun(data)
      if (!response.ok || data.status !== 'complete') notify(data.error || 'Run failed. Check the output panel.')
    } catch (error) { setRun({ run_id: '', status: 'unavailable', error: error instanceof Error ? error.message : 'API unavailable' }); notify('Could not reach the challenge API.') }
    finally { setBusy(null) }
  }

  async function submitSolution() {
    if (!run?.run_id || run.status !== 'complete') return
    setBusy('submit'); setTab('submission')
    try {
      const response = await fetch('/api/submissions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ run_id: run.run_id, session_id: sessionId }) })
      const data = await response.json()
      if (!response.ok) { notify(data.detail || 'Submission failed.'); return }
      setGrade(data as Grade)
      setChallenge((current) => current ? { ...current, explanation: data.explanation } : current)
      if (data.score >= 80) setCompletedChallengeIds((current) => new Set(current).add(selectedChallengeId))
      notify(`Submission graded · ${data.score}/100`)
    } catch { notify('Could not reach the grading API.') }
    finally { setBusy(null) }
  }

  function resetCode() { setCode(challenge?.starter_code || STARTER); setRun(null); setGrade(null); setTab('output') }
  function selectChallenge(id: string) { if (id === selectedChallengeId) return; setSelectedChallengeId(id); setChallenge(null); setRun(null); setGrade(null); setHintIndex(-1); setTab('output') }
  async function openRunHistory() {
    try {
      const response = await fetch(`/api/sessions/${sessionId}/runs`)
      if (!response.ok) throw new Error('Could not load run history')
      setRunHistory(await response.json())
      setHistoryOpen(true)
    } catch { notify('Could not load run history.') }
  }
  function nextHint() {
    if (!challenge) return
    const next = (hintIndex + 1) % challenge.hints.length
    setHintIndex(next)
    notify(`Hint ${next + 1}: ${challenge.hints[next]}`)
  }

  const attentionTrace = run?.traces?.attention
  const challengeIndex = Math.max(0, challenges.findIndex((item) => item.id === selectedChallengeId))

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark">⌁</span> circuit<span className="brand-light"> / lab</span></div>
      <div className="breadcrumbs">Challenges <span>/</span> {challenge?.category || 'Foundations'} <span>/</span> <b>{String(challengeIndex + 1).padStart(2, '0')}</b></div>
      <div className="top-actions"><span className={`api-pill ${apiState}`}><i />{apiState === 'ready' ? 'API CONNECTED' : apiState === 'loading' ? 'CONNECTING' : 'API OFFLINE'}</span><span className="checkpoint">CHECKPOINT · {challenge?.model.name.toUpperCase() || 'LOADING'}</span><button className="round-help" title="How this lab works"><CircleHelp size={15} /></button></div>
    </header>

    <div className="main-grid">
      <aside className="sidebar">
        <div className="side-label">WORKSPACE</div>
        <button className="side-link selected"><FlaskConical size={15} /> Challenge lab</button>
        <button className="side-link" onClick={openRunHistory}><Activity size={15} /> Run history</button>
        <div className="side-label path-label">{challenge?.category.toUpperCase() || 'CHALLENGES'} <span>{String(challengeIndex + 1).padStart(2, '0')} / {String(challenges.length).padStart(2, '0')}</span></div>
        {challenges.map((item, index) => <button key={item.id} className={`challenge-link ${item.id === selectedChallengeId ? 'active' : ''} ${completedChallengeIds.has(item.id) ? 'done' : ''}`} onClick={() => selectChallenge(item.id)}>
          <span className="challenge-number">{completedChallengeIds.has(item.id) ? '✓' : String(index + 1).padStart(2, '0')}</span><span className="challenge-name">{item.title}</span><span className={`level ${item.difficulty === 'Medium' ? 'med' : item.difficulty.toLowerCase()}`}>{item.difficulty.toUpperCase()}</span>
        </button>)}
        <div className="path-card"><div><BookOpen size={14} /><span>Learning path</span></div><b>{completedChallengeIds.size} of {challenges.length} complete</b><div className="path-track"><i style={{ width: `${challenges.length ? (completedChallengeIds.size / challenges.length) * 100 : 0}%` }} /></div></div>
        <div className="profile"><div className="profile-avatar">R</div><div><b>Researcher</b><small>Anonymous session</small></div><ChevronDown size={13} /></div>
      </aside>

      <section className="problem-pane">
        <div className="problem-meta"><span className="difficulty">{challenge?.difficulty.toUpperCase()} · {challenge?.category.toUpperCase()}</span><span>Problem {String(challengeIndex + 1).padStart(2, '0')}</span></div>
        <h1>{challenge?.title || 'Find the previous-token head'}</h1>
        <p className="problem-lede">{challenge?.description || 'Loading challenge…'}</p>
        <div className="problem-tabs"><button className="active">Description</button><button onClick={nextHint}><Lightbulb size={13} /> Hint {hintIndex >= 0 && <span>{hintIndex + 1}/{challenge?.hints.length}</span>}</button></div>
        <section className="statement-section"><h2>Behavior</h2><p>{challenge?.behavior}</p><div className="behavior-callout"><b>Your task</b><span>{challenge?.task_goal || 'Find the mechanism behind this behavior and support your claim with causal evidence.'}</span></div></section>
        <section className="statement-section"><h2>Examples <span>PUBLIC</span></h2><p>Examples shown here are public. The grader uses held-out sequences when you submit.</p>
          <div className="example-list">{challenge?.examples.map((example, index) => <div className="example-card" key={index}><div className="example-label">EXAMPLE {index + 1}</div><div className="example-line"><span>Input</span><code>{example.input.join(' ')}</code></div><div className="example-line output"><span>Output</span><code>{example.output.map((token) => token ?? '—').join(' ')}</code></div></div>)}</div>
          <div className="example-note"><Activity size={13} /> {challenge?.example_note || 'The first position has no previous token and is excluded from the accuracy metric.'}</div>
        </section>
        <section className="statement-section"><h2>Submission contract</h2><p>Return only the structured answer fields below. The grader runs its own causal and held-out checks; you do not submit prose or evidence claims.</p>
          <div className="contract-list">{challenge?.grading_contract.map((item) => <article className="contract-item" key={item.field}><div><code>{item.field}</code><span className="contract-type">{item.type}</span><b className={item.graded ? 'graded' : 'ungraded'}>{item.graded ? 'GRADED' : 'NOT GRADED'}</b></div><p>{item.checks}</p></article>)}</div>
          <div className="submission-format"><Braces size={14} /><code>solve(model, tokenizer, dataset, tools) → dict</code></div>
        </section>
        <section className="statement-section objectives"><h2>Learning objectives</h2>{(challenge?.learning_objectives || []).map((objective) => <div className="objective" key={objective}><Check size={13} />{objective}</div>)}</section>
        <section className="statement-section tool-reference"><h2>Model and experiment API</h2><p><code>model</code> is the challenge’s PyTorch model (wrapped to add named trace access); you can call it and inspect its tensors. Use <code>tools</code> for interventions you want recorded as grading evidence; its helpers measure effects and log experiment events.</p>
          <pre className="api-example">{`tokens = dataset[0]["tokens"]
logits = model(tokens)                 # PyTorch tensor
trace = model.run_with_trace(tokens)   # logits, attention, residuals
changed = model(tokens, ablate=(2, 1)) # forward pass with a head ablated
event = tools.ablate_head(tokens, 2, 1) # same test + recorded grading event`}</pre>
          <div className="tool-reference-list">{(challenge?.tool_reference || []).map((tool) => <article key={tool.name}><div><code>{tool.name}</code><span>{tool.type}</span></div><p>{tool.description}</p></article>)}</div>
        </section>
        <div className="metric-note"><span>BEHAVIOR METRIC</span><b>{challenge?.metric || 'Previous-token accuracy'}</b></div>
      </section>

      <section className="lab-pane">
        <div className="editor-section">
          <div className="editor-toolbar"><div className="file-tab"><Code2 size={14} /><span>solution.py</span></div><span className="function-signature">solve(model, tokenizer, dataset, tools)</span><div className="editor-buttons"><button className="reset-button" onClick={resetCode} disabled={busy !== null}><RotateCcw size={13} /> Reset</button><button className="run-button" onClick={runCode} disabled={busy !== null || apiState !== 'ready'}>{busy === 'run' ? <LoaderCircle className="spin" size={14} /> : <Play size={13} fill="currentColor" />} {busy === 'run' ? 'Running…' : 'Run code'}</button><button className="submit-button" onClick={submitSolution} disabled={!run || run.status !== 'complete' || busy !== null}>{busy === 'submit' ? <LoaderCircle className="spin" size={14} /> : <Send size={13} />} Submit</button></div></div>
          <div className="editor"><Editor height="100%" language="python" theme="vs-dark" value={code} onChange={(value) => setCode(value || '')} options={{ minimap: { enabled: false }, fontSize: 13, lineHeight: 21, tabSize: 4, insertSpaces: true, scrollBeyondLastLine: false, automaticLayout: true, padding: { top: 18 }, wordWrap: 'off', renderLineHighlight: 'line', guides: { indentation: true } }} /></div>
          <div className="editor-status"><span><i /> Python 3 · local process</span><span>UTF-8 <b>·</b> Spaces: 4</span></div>
        </div>

        <div className="result-section">
          <div className="result-toolbar"><div className="result-tabs">{(['output', 'attention', 'value_path', 'patching', 'submission'] as const).map((view) => <button key={view} className={tab === view ? 'active' : ''} onClick={() => setTab(view)}>{view === 'output' ? <Terminal size={13} /> : view === 'attention' ? <Waves size={13} /> : view === 'patching' ? <GitBranch size={13} /> : view === 'submission' ? <Send size={12} /> : <Activity size={13} />}{view.replace('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())}</button>)}</div><span className={`result-state ${run?.status || 'ready'}`}>{busy === 'run' ? 'RUNNING' : grade ? `SCORE · ${grade.score}/100` : run?.status?.toUpperCase() || 'READY'}</span></div>
          <div className="result-content">
            {tab === 'output' && <OutputPanel run={run} />}
            {tab === 'attention' && <AttentionPanel trace={attentionTrace} />}
            {tab === 'value_path' && <AnalysisPanel title="OV / VALUE PATH" data={run?.traces?.value_path} empty="Call tools.value_path(...) to trace a head's value and output contribution." />}
            {tab === 'patching' && <AnalysisPanel title="ACTIVATION PATCH" data={run?.traces?.patching} empty="Call tools.patch_head_output(...) to measure clean-to-corrupted recovery." />}
            {tab === 'submission' && <SubmissionPanel run={run} grade={grade} challenge={challenge} />}
          </div>
          <div className="result-footer"><span><Clock3 size={12} /> Runs are saved to your anonymous session</span><span>{busy ? 'PROCESSING' : 'LOCAL PYTHON WORKER'}</span></div>
        </div>
      </section>
    </div>
    {historyOpen && <RunHistoryPanel items={runHistory} challenges={challenges} onClose={() => setHistoryOpen(false)} />}
    {toast && <div className="toast"><X size={13} />{toast}</div>}
  </div>
}

function OutputPanel({ run }: { run: RunResult | null }) {
  if (!run) return <div className="empty-state"><div className="empty-icon"><Play size={17} /></div><b>Run your experiment</b><span>Print diagnostics, inspect attention, and test a causal intervention from your Python solution.</span></div>
  if (run.status !== 'complete') return <div className="execution-error"><b>Execution {run.status === 'unavailable' ? 'unavailable' : 'failed'}</b><pre>{run.error}{run.stderr ? `\n${run.stderr}` : ''}{run.traceback ? `\n${run.traceback}` : ''}</pre></div>
  return <><div className="console-label">STANDARD OUTPUT</div><pre className="console-output">{run.stdout || '(no output printed)'}</pre><div className="metric-row"><div className="metric-card"><span>BASELINE ACCURACY</span><b>{((run.baseline?.accuracy || 0) * 100).toFixed(1)}%</b></div><div className="metric-card"><span>EXPERIMENTS</span><b>{run.events?.length || 0}</b></div><div className="metric-card"><span>CLAIMED COMPONENTS</span><b>{run.submission?.components?.length || 0}</b></div></div>{run.events && run.events.length > 0 && <div className="experiment-timeline"><div className="console-label">EXPERIMENT TIMELINE</div>{run.events.map((event, index) => <div className="timeline-event" key={index}><i /><span>{event.type === 'ablation' ? `Ablated L${event.layer}H${event.head} · drop ${(Number(event.drop || 0) * 100).toFixed(0)}%` : event.type === 'activation_patch' ? `Patched L${event.layer}H${event.head} · recovery ${Number(event.recovery || 0).toFixed(2)}` : event.type === 'ov_analysis' ? `Traced OV at L${event.layer}H${event.head} · logit +${Number(event.target_logit_contribution || 0).toFixed(2)}` : event.type === 'attention' ? `Inspected attention L${event.layer}H${event.head}` : String(event.type)}</span><small>v{event.event_version || 1}</small></div>)}</div>}{run.stderr && <pre className="stderr">{run.stderr}</pre>}</>
}

function AttentionPanel({ trace }: { trace?: { layer: number; head: number; tokens: string[]; weights: number[][] } }) {
  const weights = trace?.weights
  const tokens = trace?.tokens
  if (!weights || !tokens) return <div className="empty-state compact"><div className="empty-icon"><Waves size={17} /></div><b>No attention trace yet</b><span>Call <code>tools.attention(tokens, layer, head)</code> in your code and run it.</span></div>
  return <div className="attention-wrap"><div className="attention-heading"><span>ATTENTION · LAYER {trace.layer} · HEAD {trace.head}</span><span>QUERY ↓ / KEY →</span></div><div className="attention-grid" style={{ gridTemplateColumns: `80px repeat(${tokens.length}, minmax(28px, 1fr))` }}><span /><>{tokens.map((token, i) => <span className="axis" key={`col-${i}`}>{token}</span>)}</>{weights.map((row, i) => <Fragment key={`row-${i}`}><span className="axis row-axis">{tokens[i]}</span>{row.map((value, j) => <span key={`${i}-${j}`} className="attention-cell" title={`${tokens[i]} → ${tokens[j]}: ${value.toFixed(3)}`} style={{ backgroundColor: `rgba(166, 213, 99, ${Math.max(.04, Math.min(.94, value))})` }} />)}</Fragment>)}</div></div>
}

function AnalysisPanel({ title, data, empty }: { title: string; data?: Record<string, any>; empty: string }) {
  if (!data) return <div className="empty-state compact"><div className="empty-icon"><GitBranch size={17} /></div><b>No experiment recorded</b><span>{empty}</span></div>
  return <div className="analysis-panel"><div className="console-label">{title}</div>{Object.entries(data).map(([key, value]) => <div className="analysis-row" key={key}><span>{key.replaceAll('_', ' ')}</span><b>{typeof value === 'number' ? Number(value.toFixed(4)) : String(value)}</b></div>)}</div>
}

function SubmissionPanel({ run, grade, challenge }: { run: RunResult | null; grade: Grade | null; challenge: Challenge | null }) {
  if (!run || run.status !== 'complete') return <div className="empty-state compact"><div className="empty-icon"><Braces size={17} /></div><b>Nothing submitted yet</b><span>Return the required structured component claim from <code>solve(...)</code>, then use Submit.</span></div>
  return <div className="submission-view">{grade && <><div className="grade-heading"><div><span>CHALLENGE SCORE</span><b>{grade.score}<small>/100</small></b></div><div className="grade-breakdown">{Object.entries(grade.dimensions).map(([name, score]) => <div key={name}><span>{name.replaceAll('_', ' ')}</span><b>{score}</b></div>)}</div></div><div className="feedback-list">{grade.feedback.map((item) => <p key={item}><Check size={13} />{item}</p>)}</div>{challenge?.explanation?.canonical_solution && <div className="explanation-card"><b>Mechanism</b><p>{challenge.explanation.canonical_solution}</p>{challenge.explanation.caution && <small>{challenge.explanation.caution}</small>}</div>}</>}<div className="console-label">STRUCTURED CLAIM</div><pre className="claim-json">{JSON.stringify(run.submission || {}, null, 2)}</pre>{!grade && <div className="submit-help">Run code, then press <b>Submit</b> to evaluate this claim against the held-out challenge set.</div>}</div>
}

function RunHistoryPanel({ items, challenges, onClose }: { items: HistoryRun[]; challenges: ChallengeSummary[]; onClose: () => void }) {
  const titles = new Map(challenges.map((challenge) => [challenge.id, challenge.title]))
  return <div className="modal-backdrop" onClick={onClose}><section className="history-modal" onClick={(event) => event.stopPropagation()}><header><div><b>Run history</b><small>Recent experiments saved for this browser session</small></div><button onClick={onClose} aria-label="Close run history"><X size={16} /></button></header>{items.length ? items.map((item) => <article key={item.run_id}><div><b>{titles.get(item.challenge_id) || item.challenge_id}</b><small>{item.created_at} · {item.run_id.slice(0, 8)}</small></div><span className={`history-status ${item.status}`}>{item.status}</span></article>) : <div className="history-empty">No runs yet. Run an experiment to see it here.</div>}</section></div>
}

export default App
