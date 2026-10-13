import { useEffect, useMemo, useState } from 'react'
import Editor from '@monaco-editor/react'
import { Activity, BookOpen, Braces, Check, CircleHelp, Clock3, Code2, FlaskConical, Lightbulb, LoaderCircle, Play, RotateCcw, Send, Terminal, X } from 'lucide-react'

type Example = { input: string[]; output: (string | null)[] }
type Challenge = {
  id: string; title: string; difficulty: string; category: string; description: string; behavior: string; metric: string
  model: { name: string; layers: number; heads: number; d_model: number }
  examples: Example[]; learning_objectives: string[]; tools: string[]; hints: string[]; starter_code?: string; task_goal?: string; example_note?: string
  milestones?: Array<{ id: string; title: string; description: string; verification: string }>
  grading_contract: Array<{ field: string; type: string; graded: boolean; checks: string }>
  tool_reference: Array<{ name: string; type: string; description: string }>
  explanation?: { canonical_solution?: string; caution?: string }
}
type ChallengeSummary = Pick<Challenge, 'id' | 'title' | 'difficulty' | 'category'>
type HistoryRun = { run_id: string; challenge_id: string; status: string; created_at: string; source?: string; result?: RunResult | null }
type RunResult = {
  run_id: string; status: 'complete' | 'error' | 'unavailable'; stdout?: string; stderr?: string; error?: string
  artifacts?: Array<{ name: string; mime_type: string; data: string }>
  traceback?: string; events?: Array<Record<string, any>>; baseline?: { accuracy: number; correct: number; total: number }
  submission?: Record<string, any>
  graded_milestone?: string; grade?: Grade
}
type MissionProgress = { milestones: Array<{ id: string; passed: boolean; unlocked: boolean; points: number }>; earned_points: number; possible_points: number; next_milestone_id: string | null; completed_independently: boolean; solution_revealed: boolean; status: string }
type Grade = { submission_id: string; score: number; earned_points?: number; possible_points?: number; milestone_score?: number; milestone_id?: string; passed?: boolean; checks: Record<string, any>; feedback: string[]; progress?: MissionProgress; explanation?: Challenge['explanation'] }

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
  const [selectedChallengeId, setSelectedChallengeId] = useState('palindrome-machine')
  const [apiState, setApiState] = useState<'loading' | 'ready' | 'offline'>('loading')
  const [code, setCode] = useState('')
  const [run, setRun] = useState<RunResult | null>(null)
  const [grade, setGrade] = useState<Grade | null>(null)
  const [missionProgress, setMissionProgress] = useState<MissionProgress | null>(null)
  const [reference, setReference] = useState<Challenge['explanation'] | null>(null)
  const [busy, setBusy] = useState<'run' | 'submit' | null>(null)
  const [tab, setTab] = useState<'output' | 'submission'>('output')
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
        const progressResponse = await fetch(`/api/missions/${selectedChallengeId}/progress?session_id=${sessionId}`).catch(() => null)
        const progressData = progressResponse?.ok ? await progressResponse.json() as MissionProgress : null
        if (active) {
          setChallenge(data); setCode(draft?.source || data.starter_code || ''); setApiState('ready')
          setMissionProgress(progressData); setReference(null)
          if (progressData?.completed_independently) setCompletedChallengeIds((current) => new Set(current).add(selectedChallengeId))
        }
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
      const response = await fetch('/api/submissions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ run_id: run.run_id, session_id: sessionId, milestone_id: missionProgress?.next_milestone_id }) })
      const data = await response.json()
      if (!response.ok) { notify(data.detail || 'Submission failed.'); return }
      setGrade(data as Grade)
      if (data.progress) setMissionProgress(data.progress)
      if (data.explanation) setReference(data.explanation)
      if (data.progress?.completed_independently) setCompletedChallengeIds((current) => new Set(current).add(selectedChallengeId))
      notify(data.passed ? `Milestone complete · ${data.progress?.earned_points ?? data.score}/100 points` : `Milestone feedback · ${data.score}/${data.possible_points ?? 100} points`)
    } catch { notify('Could not reach the grading API.') }
    finally { setBusy(null) }
  }

  async function revealSolution() {
    try {
      const response = await fetch(`/api/missions/${selectedChallengeId}/reveal`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ session_id: sessionId }) })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not reveal the reference investigation')
      setMissionProgress(data.progress); setReference(data.explanation)
      notify('Reference investigation revealed · this mission is marked solution revealed')
    } catch (error) { notify(error instanceof Error ? error.message : 'Could not reveal the solution.') }
  }

  function resetCode() { setCode(challenge?.starter_code || ''); setRun(null); setGrade(null); setTab('output') }
  function selectChallenge(id: string) { if (id === selectedChallengeId) return; setSelectedChallengeId(id); setChallenge(null); setRun(null); setGrade(null); setMissionProgress(null); setReference(null); setHintIndex(-1); setTab('output') }
  async function openRunHistory() {
    try {
      const response = await fetch(`/api/sessions/${sessionId}/runs`)
      if (!response.ok) throw new Error('Could not load run history')
      setRunHistory(await response.json())
      setHistoryOpen(true)
    } catch { notify('Could not load run history.') }
  }
  function restoreRun(item: HistoryRun) {
    if (item.challenge_id !== selectedChallengeId) selectChallenge(item.challenge_id)
    if (item.source !== undefined) setCode(item.source)
    if (item.result) setRun({ ...item.result, run_id: item.run_id } as RunResult)
    setGrade(null); setTab('output'); setHistoryOpen(false)
  }
  function nextHint() {
    if (!challenge) return
    const next = (hintIndex + 1) % challenge.hints.length
    setHintIndex(next)
    notify(`Hint ${next + 1}: ${challenge.hints[next]}`)
  }

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
      </aside>

      <section className="problem-pane">
        <div className="problem-meta"><span className="difficulty">{challenge?.difficulty.toUpperCase()} · {challenge?.category.toUpperCase()}</span><span>Problem {String(challengeIndex + 1).padStart(2, '0')}</span></div>
        <h1>{challenge?.title || 'The Palindrome Machine'}</h1>
        <p className="problem-lede">{challenge?.description || 'Loading challenge…'}</p>
        <div className="problem-tabs"><button className="active">Description</button><button onClick={nextHint}><Lightbulb size={13} /> Hint {hintIndex >= 0 && <span>{hintIndex + 1}/{challenge?.hints.length}</span>}</button></div>
        <section className="statement-section"><h2>Behavior</h2><p>{challenge?.behavior}</p><div className="behavior-callout"><b>Your task</b><span>{challenge?.task_goal || 'Find the mechanism behind this behavior and support your claim with causal evidence.'}</span></div></section>
        {challenge?.milestones?.length ? <section className="statement-section"><h2>Research milestones <span>{missionProgress?.earned_points ?? 0}/100</span></h2><div className="milestone-list">{challenge.milestones.map((milestone, index) => { const state = missionProgress?.milestones.find((item) => item.id === milestone.id); return <article key={milestone.id} className={`${state?.passed ? 'passed' : ''} ${state?.unlocked ? 'unlocked' : 'locked'}`}><span>{state?.passed ? '✓' : String(index + 1).padStart(2, '0')}</span><div><b>{milestone.title}</b><p>{milestone.description}</p><small>{state?.passed ? 'PASSED' : state?.unlocked ? 'UNLOCKED · ' + milestone.verification : 'LOCKED · complete the previous milestone'}</small></div></article> })}</div>{missionProgress?.status === 'solution_revealed' && <div className="solution-status">Reference revealed · progress is recorded as solution revealed, not independent completion.</div>}</section> : null}
        <section className="statement-section"><h2>Examples <span>PUBLIC</span></h2><p>Examples shown here are public. The grader uses held-out sequences when you submit.</p>
          <div className="example-list">{challenge?.examples.map((example, index) => <div className="example-card" key={index}><div className="example-label">EXAMPLE {index + 1}</div><div className="example-line"><span>Input</span><code>{example.input.join(' ')}</code></div><div className="example-line output"><span>Output</span><code>{example.output.map((token) => token ?? '—').join(' ')}</code></div></div>)}</div>
          <div className="example-note"><Activity size={13} /> {challenge?.example_note || 'The first position has no previous token and is excluded from the accuracy metric.'}</div>
        </section>
        <section className="statement-section"><h2>Research submission</h2><p>Submit evidence for the currently unlocked milestone. Hidden evaluation reruns your named research functions and independently replays interventions; printed scores are ignored.</p>
          <div className="contract-list">{challenge?.grading_contract.map((item) => <article className="contract-item" key={item.field}><div><code>{item.field}</code><span className="contract-type">{item.type}</span><b className={item.graded ? 'graded' : 'ungraded'}>{item.graded ? 'GRADED' : 'NOT GRADED'}</b></div><p>{item.checks}</p></article>)}</div>
          <div className="submission-format"><Braces size={14} /><code>Unlocked: {challenge?.milestones?.find((item) => item.id === missionProgress?.next_milestone_id)?.title || (missionProgress?.completed_independently ? 'Mission complete' : 'loading progress')}</code></div>
          <button className="reveal-button" onClick={revealSolution} disabled={missionProgress?.solution_revealed || missionProgress?.completed_independently}>{missionProgress?.solution_revealed ? 'Reference already revealed' : missionProgress?.completed_independently ? 'Mission independently completed' : 'Reveal reference investigation'}</button>
        </section>
        <section className="statement-section objectives"><h2>Learning objectives</h2>{(challenge?.learning_objectives || []).map((objective) => <div className="objective" key={objective}><Check size={13} />{objective}</div>)}</section>
        <section className="statement-section tool-reference"><h2>Python research API</h2><p><code>model</code> is the trained PyTorch model. Inspect raw tensors and write your own measurements and plots. Use <code>tools</code> when you want a replayable intervention recorded with the run.</p>
          <pre className="api-example">{`tokens = dataset[0]["tokens"]
logits, cache = model.run_with_cache(tokens)
attention = model.get_attention(tokens, layer=0, head=0)
print(cache.keys(), attention.shape)
plt.imshow(attention.detach().cpu())   # your code creates the figure
plt.show()`}</pre>
          <div className="tool-reference-list">{(challenge?.tool_reference || []).map((tool) => <article key={tool.name}><div><code>{tool.name}</code><span>{tool.type}</span></div><p>{tool.description}</p></article>)}</div>
        </section>
        <div className="metric-note"><span>BEHAVIOR METRIC</span><b>{challenge?.metric || 'Previous-token accuracy'}</b></div>
      </section>

      <section className="lab-pane">
        <div className="editor-section">
          <div className="editor-toolbar"><div className="file-tab"><Code2 size={14} /><span>research.py</span></div><span className="function-signature">clean process per run</span><div className="editor-buttons"><button className="reset-button" onClick={resetCode} disabled={busy !== null}><RotateCcw size={13} /> Reset</button><button className="run-button" onClick={runCode} disabled={busy !== null || apiState !== 'ready'}>{busy === 'run' ? <LoaderCircle className="spin" size={14} /> : <Play size={13} fill="currentColor" />} {busy === 'run' ? 'Running…' : 'Run code'}</button><button className="submit-button" onClick={submitSolution} disabled={!run || run.status !== 'complete' || busy !== null || !missionProgress?.next_milestone_id}>{busy === 'submit' ? <LoaderCircle className="spin" size={14} /> : <Send size={13} />} Submit</button></div></div>
          <div className="editor"><Editor height="100%" language="python" theme="vs-dark" value={code} onChange={(value) => setCode(value || '')} options={{ minimap: { enabled: false }, fontSize: 13, lineHeight: 21, tabSize: 4, insertSpaces: true, scrollBeyondLastLine: false, automaticLayout: true, padding: { top: 18 }, wordWrap: 'off', renderLineHighlight: 'line', guides: { indentation: true } }} /></div>
          <div className="editor-status"><span><i /> Python 3 · local process</span><span>UTF-8 <b>·</b> Spaces: 4</span></div>
        </div>

        <div className="result-section">
          <div className="result-toolbar"><div className="result-tabs">{(['output', 'submission'] as const).map((view) => <button key={view} className={tab === view ? 'active' : ''} onClick={() => setTab(view)}>{view === 'output' ? <Terminal size={13} /> : <Send size={12} />}{view === 'output' ? 'Run output' : 'Milestone'}</button>)}</div><span className={`result-state ${run?.status || 'ready'}`}>{busy === 'run' ? 'RUNNING' : grade ? `${grade.passed ? 'PASSED' : 'INCOMPLETE'} · ${missionProgress?.earned_points ?? grade.score}/100` : run?.status?.toUpperCase() || 'READY'}</span></div>
          <div className="result-content">
            {tab === 'output' && <OutputPanel run={run} />}
            {tab === 'submission' && <SubmissionPanel run={run} grade={grade} reference={reference} />}
          </div>
          <div className="result-footer"><span><Clock3 size={12} /> Runs are saved to your anonymous session</span><span>{busy ? 'PROCESSING' : 'LOCAL PYTHON WORKER'}</span></div>
        </div>
      </section>
    </div>
    {historyOpen && <RunHistoryPanel items={runHistory} challenges={challenges} onClose={() => setHistoryOpen(false)} onSelect={restoreRun} />}
    {toast && <div className="toast"><X size={13} />{toast}</div>}
  </div>
}

function OutputPanel({ run }: { run: RunResult | null }) {
  if (!run) return <div className="empty-state"><div className="empty-icon"><Play size={17} /></div><b>Run your experiment</b><span>Print diagnostics, inspect attention, and test a causal intervention from your research script.</span></div>
  if (run.status !== 'complete') return <div className="execution-error"><b>Execution {run.status === 'unavailable' ? 'unavailable' : 'failed'}</b><pre>{run.error}{run.stderr ? `\n${run.stderr}` : ''}{run.traceback ? `\n${run.traceback}` : ''}</pre></div>
  return <>{run.submission && (run.submission.hypothesis || run.submission.expected_result || run.submission.notes) && <div className="research-note"><div className="console-label">RESEARCH JOURNAL NOTE {run.graded_milestone ? `· ${run.graded_milestone}` : ''}</div>{run.submission.hypothesis && <p><b>Hypothesis</b>{run.submission.hypothesis}</p>}{run.submission.expected_result && <p><b>Expected</b>{run.submission.expected_result}</p>}{run.submission.notes && <p><b>Observation</b>{run.submission.notes}</p>}</div>}<div className="console-label">STANDARD OUTPUT</div><pre className="console-output">{run.stdout || '(no output printed)'}</pre>{run.artifacts?.length ? <div className="run-artifacts"><div className="console-label">FIGURES FROM YOUR CODE</div>{run.artifacts.map((artifact) => <figure key={artifact.name}><img src={`data:${artifact.mime_type};base64,${artifact.data}`} alt={artifact.name} /><figcaption>{artifact.name}</figcaption></figure>)}</div> : null}<div className="metric-row"><div className="metric-card"><span>BASELINE ACCURACY</span><b>{((run.baseline?.accuracy || 0) * 100).toFixed(1)}%</b></div><div className="metric-card"><span>EXPERIMENTS</span><b>{run.events?.length || 0}</b></div><div className="metric-card"><span>CLAIMED COMPONENTS</span><b>{run.submission?.components?.length || 0}</b></div></div>{run.events && run.events.length > 0 && <div className="experiment-timeline"><div className="console-label">EXPERIMENT TIMELINE</div>{run.events.map((event, index) => <div className="timeline-event" key={index}><i /><span>{event.type === 'ablation' ? `Ablated L${event.layer}H${event.head} · drop ${(Number(event.drop || 0) * 100).toFixed(0)}%` : event.type === 'mlp_ablation' ? `Ablated MLP ${event.layer} · drop ${(Number(event.drop || 0) * 100).toFixed(0)}%` : event.type === 'activation_patch' ? `Patched L${event.layer}H${event.head} · recovery ${Number(event.recovery || 0).toFixed(2)}` : event.type === 'mlp_patch' ? `Patched MLP ${event.layer} · margin change ${Number(event.recovery || 0).toFixed(2)}` : event.type === 'residual_patch' ? `Patched residual L${event.layer} position ${event.position} · recovery ${Number(event.recovery || 0).toFixed(2)}` : event.type === 'ov_analysis' ? `Traced OV at L${event.layer}H${event.head} · logit +${Number(event.target_logit_contribution || 0).toFixed(2)}` : event.type === 'attention' ? `Inspected attention L${event.layer}H${event.head}` : String(event.type)}</span><small>v{event.event_version || 1}</small></div>)}</div>}{run.stderr && <pre className="stderr">{run.stderr}</pre>}</>
}

function SubmissionPanel({ run, grade, reference }: { run: RunResult | null; grade: Grade | null; reference: Challenge['explanation'] | null }) {
  if (!run || run.status !== 'complete') return <div className="empty-state compact"><div className="empty-icon"><Braces size={17} /></div><b>Nothing submitted yet</b><span>Run a research script, set a <code>submission</code> dictionary with your hypothesis and tested components, then press Submit.</span></div>
  return <div className="submission-view">{grade && <><div className="grade-heading"><div><span>MILESTONE</span><b>{grade.milestone_score ?? (grade.passed ? 100 : 0)}<small>/100</small></b></div><div className="grade-breakdown"><div><span>mission points</span><b>{grade.progress?.earned_points ?? grade.earned_points ?? 0} / 100</b></div><div><span>status</span><b>{grade.passed ? 'passed' : 'in progress'}</b></div><div><span>unlocked next</span><b>{grade.progress?.next_milestone_id || 'final'}</b></div></div></div><div className="feedback-list">{grade.feedback.map((item) => <p key={item}><Check size={13} />{item}</p>)}</div></>}{reference?.canonical_solution && <div className="explanation-card"><b>Reference investigation</b><p>{reference.canonical_solution}</p>{reference.caution && <small>{reference.caution}</small>}</div>}<div className="console-label">RESEARCH CLAIM</div><pre className="claim-json">{JSON.stringify(run.submission || {}, null, 2)}</pre>{!grade && <div className="submit-help">The grader runs your research functions against hidden model activations and independently replays causal interventions.</div>}</div>
}

function RunHistoryPanel({ items, challenges, onClose, onSelect }: { items: HistoryRun[]; challenges: ChallengeSummary[]; onClose: () => void; onSelect: (item: HistoryRun) => void }) {
  const titles = new Map(challenges.map((challenge) => [challenge.id, challenge.title]))
  const [compareIds, setCompareIds] = useState<string[]>([])
  const comparison = compareIds.map((id) => items.find((item) => item.run_id === id)).filter((item): item is HistoryRun => Boolean(item))
  function toggleCompare(id: string) { setCompareIds((current) => current.includes(id) ? current.filter((value) => value !== id) : [...current.slice(-1), id]) }
  return <div className="modal-backdrop" onClick={onClose}><section className="history-modal" onClick={(event) => event.stopPropagation()}><header><div><b>Research journal</b><small>Restore or compare experiments · source, notes, output, and figures are kept</small></div><button onClick={onClose} aria-label="Close run history"><X size={16} /></button></header>{items.length ? items.map((item) => <article className="journal-row" key={item.run_id}><button className="journal-entry" onClick={() => onSelect(item)}><div><b>{titles.get(item.challenge_id) || item.challenge_id}</b><small>{item.created_at} · {item.run_id.slice(0, 8)}{item.result?.graded_milestone ? ` · ${item.result.graded_milestone}` : ''}</small></div><span className={`history-status ${item.status}`}>{item.status}</span></button><label><input type="checkbox" checked={compareIds.includes(item.run_id)} onChange={() => toggleCompare(item.run_id)} />Compare</label></article>) : <div className="history-empty">No experiments yet. Run an investigation to see it here.</div>}{comparison.length === 2 && <div className="run-comparison"><div className="console-label">SIDE-BY-SIDE EXPERIMENTS</div><div className="comparison-grid">{comparison.map((item) => <article key={item.run_id}><b>{item.result?.graded_milestone || 'Exploration'}</b><small>{item.run_id.slice(0, 8)}</small><p><strong>Hypothesis</strong>{item.result?.submission?.hypothesis || '—'}</p><p><strong>Expected</strong>{item.result?.submission?.expected_result || '—'}</p><p><strong>Notes</strong>{item.result?.submission?.notes || '—'}</p><pre>{item.result?.stdout || '(no output)'}</pre></article>)}</div></div>}</section></div>
}

export default App
