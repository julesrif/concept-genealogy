<!--
  RunView.vue — Live pipeline monitor with HITL gate interaction.

  Design decisions:
  - Polling (setInterval 2 s) chosen over WebSockets because: (a) the LangGraph pipeline
    advances in discrete steps with multi-second latency between stages, making push
    unnecessary; (b) polling is simpler to implement behind reverse proxies and load
    balancers; (c) reconnection logic for WebSocket adds complexity disproportionate
    to the update frequency (Russell & Norvig, 2021 — simplicity principle).
  - Pipeline progress bar renders all stages declaratively from a static list, with
    CSS classes toggled by comparing the current run stage — avoids imperative DOM
    manipulation and leverages Vue's reactive rendering.
  - HITL gate cards render conditionally (v-if on gate_type) so only the active gate
    is shown; this prevents user confusion and enforces the sequential gate protocol
    defined in the orchestrator (Mosqueira-Rey et al., 2023).
  - Gate approval/rejection POSTs update server state and then re-poll immediately,
    ensuring the UI reflects the new pipeline state without waiting for the next
    polling interval.
-->
<template>
  <div class="run-view">
    <div v-if="!run" class="card loading-card">
      <p>Loading run...</p>
    </div>

    <template v-else>
      <!-- Header -->
      <div class="run-header card">
        <div class="run-header-top">
          <h2>{{ run.user_concept }}</h2>
          <span :class="'badge badge-' + statusClass(run.status)">{{ run.status }}</span>
        </div>
        <p class="run-meta">Run ID: {{ run.run_id }} | Stage: {{ run.stage }}</p>
      </div>

      <!-- Pipeline progress -->
      <div class="pipeline-progress card">
        <h3>Pipeline Progress</h3>
        <div class="stages">
          <div
            v-for="stage in stages"
            :key="stage.key"
            :class="['stage-item', { active: isStageActive(stage.key), done: isStageDone(stage.key) }]"
          >
            <div class="stage-dot"></div>
            <span>{{ stage.label }}</span>
          </div>
        </div>
      </div>

      <!-- HITL Gate 1: Seed Selection -->
      <div v-if="run.current_gate?.gate_type === 'seed_selection'" class="card gate-card">
        <h3>🎯 Gate 1: Select Seed Paper</h3>
        <p class="gate-desc">
          Choose a paper to use as the starting point for citation tracing.
        </p>
        <div class="candidates">
          <div
            v-for="(paper, idx) in run.current_gate.candidates"
            :key="paper.paper_id"
            :class="['candidate', { selected: selectedIdx === idx }]"
            @click="selectedIdx = idx"
          >
            <div class="candidate-header">
              <span class="candidate-num">{{ idx + 1 }}</span>
              <h4>{{ paper.title }}</h4>
            </div>
            <div class="candidate-meta">
              <span>{{ (paper.authors || []).slice(0, 3).join(', ') }}{{ paper.authors?.length > 3 ? ' et al.' : '' }}</span>
              <span>Year: {{ paper.year || 'N/A' }}</span>
              <span>Citations: {{ paper.citation_count || 0 }}</span>
              <span v-if="paper.sjr_score" class="sjr-score">SJR: {{ paper.sjr_score.toFixed(2) }}</span>
              <span :class="paper.open_access_url ? 'oa-yes' : paper.doi ? 'oa-elsevier' : 'oa-no'">
                {{ paper.open_access_url ? '✓ Open Access' : paper.doi ? '✓ Elsevier (via DOI)' : '✗ Abstract only' }}
              </span>
              <a
                v-if="paper.doi"
                :href="`https://doi.org/${paper.doi}`"
                target="_blank"
                rel="noopener"
                class="doi-link"
                @click.stop
              >
                DOI ↗
              </a>
            </div>
            <p v-if="paper.abstract" class="candidate-abstract">
              {{ paper.abstract.slice(0, 250) }}{{ paper.abstract.length > 250 ? '...' : '' }}
            </p>
          </div>
        </div>
        <div class="gate-actions">
          <button
            class="btn btn-success"
            :disabled="selectedIdx === null || responding"
            @click="approveGate1"
          >
            [OK] Select Paper {{ selectedIdx !== null ? selectedIdx + 1 : '' }}
          </button>
          <button class="btn btn-danger" :disabled="responding" @click="rejectGate">
            ✗ Reject All (New Search)
          </button>
        </div>
      </div>

      <!-- HITL Concept Selection Gate (DOI mode) -->
      <div v-if="run.current_gate?.gate_type === 'concept_selection'" class="card gate-card">
        <h3>🔬 Select Concepts to Trace</h3>
        <p class="gate-desc">
          The following concepts were extracted from the paper.
          Select up to 3 concepts to trace through citation chains.
        </p>
        <div class="concept-list">
          <div
            v-for="(concept, idx) in run.current_gate.concepts"
            :key="idx"
            :class="['concept-item', { selected: selectedConceptIndices.includes(idx) }]"
            @click="toggleConcept(idx)"
          >
            <span class="concept-check">{{ selectedConceptIndices.includes(idx) ? '☑' : '☐' }}</span>
            <span class="concept-text">{{ concept }}</span>
          </div>
        </div>
        <p class="concept-count">
          {{ selectedConceptIndices.length }}/3 selected
        </p>
        <div class="gate-actions">
          <button
            class="btn btn-success"
            :disabled="selectedConceptIndices.length === 0 || responding"
            @click="approveConceptSelection"
          >
            [OK] Trace Selected Concepts
          </button>
          <button class="btn btn-danger" :disabled="responding" @click="rejectGate">
            ✗ Cancel
          </button>
        </div>
      </div>

      <!-- Tracing Progress (visible during tracing stage) -->
      <div v-if="run.stage === 'tracing' && tracingProgress.currentDepth >= 0" class="card tracing-progress-card">
        <h3>🔍 Tracing Citations</h3>
        <div class="tracing-stats">
          <div class="tracing-stat">
            <span class="stat-label">Depth</span>
            <span class="stat-value">{{ tracingProgress.currentDepth }} / {{ tracingProgress.maxDepth }}</span>
          </div>
          <div class="tracing-stat">
            <span class="stat-label">Papers traced</span>
            <span class="stat-value">{{ tracingProgress.papersTraced }}</span>
          </div>
          <div class="tracing-stat">
            <span class="stat-label">Claims found</span>
            <span class="stat-value">{{ tracingProgress.totalClaims }}</span>
          </div>
          <div class="tracing-stat">
            <span class="stat-label">References found</span>
            <span class="stat-value">{{ tracingProgress.totalRefs }}</span>
          </div>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar" :style="{ width: tracingProgress.percent + '%' }"></div>
        </div>
        <p class="tracing-current" v-if="tracingProgress.currentPaper">
          Currently: {{ tracingProgress.currentPaper }}
        </p>
        <div v-if="tracingProgress.conceptAbstractions.length" class="concept-abstractions">
          <span class="stat-label">Concept evolution:</span>
          <div class="concept-chain">
            <span
              v-for="(abs, idx) in tracingProgress.conceptAbstractions"
              :key="idx"
              class="concept-step"
            >
              <span class="concept-name">{{ abs.fromConcept }}</span>
              <span class="concept-arrow">→</span>
            </span>
            <span class="concept-name concept-current">
              {{ tracingProgress.conceptAbstractions[tracingProgress.conceptAbstractions.length - 1].toConcept }}
            </span>
          </div>
        </div>
      </div>

      <!-- HITL Graph Gate: Review genealogy graph before synthesis -->
      <div v-if="run.current_gate?.gate_type === 'graph_review'" class="card gate-card">
        <h3>🌳 Graph Review: Concept Genealogy</h3>
        <p class="gate-desc">
          Review the traced citation chain below. Nodes are colour-coded by depth level.
          Approve to proceed with synthesis, or reject to stop.
        </p>
        <GenealogyGraph v-if="graphData" :graphData="graphData" />
        <p v-else class="graph-loading">Loading graph data...</p>
        <div class="gate-actions">
          <button class="btn btn-success" :disabled="responding || !graphData" @click="approveGate">
            [OK] Approve Graph
          </button>
          <button class="btn btn-danger" :disabled="responding" @click="rejectGate">
            ✗ Reject (Stop Pipeline)
          </button>
        </div>
      </div>

      <!-- HITL Gate 2: Synthesis Approval -->
      <div v-if="run.current_gate?.gate_type === 'synthesis_approval'" class="card gate-card">
        <h3>📝 Gate 2: Review Synthesis</h3>
        <p class="gate-desc">
          Review the generated synthesis. Approve to proceed to adversarial review,
          or reject to trigger re-synthesis.
        </p>
        <div class="synthesis-content">
          <pre>{{ run.current_gate.synthesis_text || '(empty synthesis)' }}</pre>
        </div>
        <div class="gate-actions">
          <button class="btn btn-success" :disabled="responding" @click="approveGate">
            [OK] Approve Synthesis
          </button>
          <button class="btn btn-danger" :disabled="responding" @click="rejectGate">
            ✗ Reject (Re-synthesise)
          </button>
        </div>
      </div>

      <!-- HITL Gate 3: Final Approval -->
      <div v-if="run.current_gate?.gate_type === 'final_approval'" class="card gate-card">
        <h3>🏁 Gate 3: Final Approval</h3>
        <p class="gate-desc">
          Review the final output including the Challenger's critique.
          Approving will persist results to the database.
        </p>
        <div class="synthesis-content">
          <h4>Synthesis</h4>
          <pre>{{ run.current_gate.synthesis_text || '(no synthesis)' }}</pre>
        </div>
        <div v-if="run.current_gate.challenger_critique" class="synthesis-content challenger">
          <h4>Challenger Review</h4>
          <pre>{{ run.current_gate.challenger_critique }}</pre>
        </div>
        <div class="gate-actions">
          <button class="btn btn-success" :disabled="responding" @click="approveGate">
            [OK] Approve & Persist
          </button>
          <button class="btn btn-danger" :disabled="responding" @click="rejectGate">
            ✗ Reject (Back to Synthesis)
          </button>
        </div>
      </div>

      <!-- Completed state -->
      <div v-if="run.status === 'completed'" class="card completed-card">
        <h3>[OK] Pipeline Complete</h3>
        <p>Output saved to: <code>{{ run.output_path }}</code></p>
        <button class="btn btn-primary" @click="viewResult">View Result</button>
      </div>

      <!-- Failed state -->
      <div v-if="run.status === 'failed'" class="card failed-card">
        <h3>✗ Pipeline Failed</h3>
        <p class="error">{{ run.error_message }}</p>
      </div>

      <!-- Live Log Panel -->
      <div class="card log-panel">
        <div class="log-header">
          <h3>📋 Pipeline Log</h3>
          <span class="log-count">{{ logs.length }} entries</span>
        </div>
        <div class="log-entries" ref="logContainer">
          <div v-if="logs.length === 0" class="log-empty">
            Waiting for pipeline output...
          </div>
          <div
            v-for="(entry, idx) in logs"
            :key="idx"
            :class="['log-entry', 'log-' + entry.level.toLowerCase()]"
          >
            <span class="log-time">{{ formatLogTime(entry.timestamp) }}</span>
            <span class="log-level">{{ entry.level }}</span>
            <span class="log-source">{{ entry.source }}</span>
            <span class="log-msg">{{ entry.message }}</span>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { getRun, respondToGate, getLogs, getGraph } from '../api.js'
import GenealogyGraph from '../components/GenealogyGraph.vue'

const props = defineProps({ runId: String })
const router = useRouter()

const run = ref(null)
const logs = ref([])
const selectedIdx = ref(null)
const selectedConceptIndices = ref([])
const responding = ref(false)
const logContainer = ref(null)
const graphData = ref(null)
let pollInterval = null
let logOffset = 0
let graphFetched = false

const stages = [
  { key: 'validating_doi', label: 'Validating DOI' },
  { key: 'extracting_concepts', label: 'Extracting Concepts' },
  { key: 'awaiting_concept_selection', label: 'Concept Selection' },
  { key: 'retrieving', label: 'Retrieving' },
  { key: 'awaiting_seed_approval', label: 'Seed Selection' },
  { key: 'tracing', label: 'Tracing Citations' },
  { key: 'awaiting_graph_approval', label: 'Graph Review' },
  { key: 'synthesising', label: 'Synthesising' },
  { key: 'awaiting_synthesis_approval', label: 'Synthesis Review' },
  { key: 'challenging', label: 'Challenger Review' },
  { key: 'awaiting_final_approval', label: 'Final Approval' },
  { key: 'persisting', label: 'Persisting' },
  { key: 'done', label: 'Done' },
]

const stageOrder = stages.map(s => s.key)

// Parse tracing progress from pipeline log messages
const tracingProgress = computed(() => {
  const progress = { currentDepth: -1, maxDepth: 3, papersTraced: 0, totalClaims: 0, totalRefs: 0, currentPaper: '', conceptAbstractions: [], percent: 0 }
  for (const entry of logs.value) {
    if (entry.source !== 'pipeline') continue
    const msg = entry.message

    // "Depth N, Paper M — "Title""
    const depthMatch = msg.match(/Depth (\d+), Paper (\d+) — "(.+)"/)
    if (depthMatch) {
      progress.currentDepth = parseInt(depthMatch[1])
      progress.papersTraced++
      progress.currentPaper = depthMatch[3]
    }

    // "Depth N — X claims, Y references found"
    const resultMatch = msg.match(/Depth \d+ — (\d+) claims, (\d+) references found/)
    if (resultMatch) {
      progress.totalClaims += parseInt(resultMatch[1])
      progress.totalRefs += parseInt(resultMatch[2])
    }

    // "Advancing to depth N: M papers to trace"
    const advanceMatch = msg.match(/Advancing to depth (\d+)/)
    if (advanceMatch) {
      progress.currentDepth = parseInt(advanceMatch[1])
    }

    // "Concept abstraction: depth N "X" → depth N+1 "Y""
    const abstractMatch = msg.match(/Concept abstraction: depth (\d+) "(.+?)" → depth (\d+) "(.+?)"/)
    if (abstractMatch) {
      progress.conceptAbstractions.push({
        fromDepth: parseInt(abstractMatch[1]),
        fromConcept: abstractMatch[2],
        toDepth: parseInt(abstractMatch[3]),
        toConcept: abstractMatch[4],
      })
    }
  }
  // Estimate percent based on depth progression
  if (progress.currentDepth >= 0) {
    progress.percent = Math.min(95, Math.round((progress.currentDepth / progress.maxDepth) * 100))
  }
  return progress
})

function isStageActive(key) {
  return run.value?.stage === key
}

function isStageDone(key) {
  const currentIdx = stageOrder.indexOf(run.value?.stage)
  const stageIdx = stageOrder.indexOf(key)
  return stageIdx < currentIdx
}

function statusClass(status) {
  const map = { running: 'running', awaiting_input: 'awaiting', completed: 'completed', failed: 'failed' }
  return map[status] || 'running'
}

async function fetchRun() {
  try {
    run.value = await getRun(props.runId)

    // Fetch graph when the graph review gate is active
    if (!graphFetched && run.value.current_gate?.gate_type === 'graph_review') {
      fetchGraph()
    }

    if (run.value.status === 'completed' || run.value.status === 'failed') {
      // Fetch final logs then stop polling
      await fetchLogs()
      if (pollInterval) clearInterval(pollInterval)
    }
  } catch {
    // retry on next poll
  }
}

async function fetchGraph() {
  try {
    const data = await getGraph(props.runId)
    if (data.nodes && data.nodes.length > 0) {
      graphData.value = data
      graphFetched = true
    }
  } catch {
    // graph not ready yet, will retry on next poll
  }
}

async function fetchLogs() {
  try {
    const newLogs = await getLogs(props.runId, logOffset)
    if (newLogs.length > 0) {
      logs.value.push(...newLogs)
      logOffset += newLogs.length
      await nextTick()
      if (logContainer.value) {
        logContainer.value.scrollTop = logContainer.value.scrollHeight
      }
    }
  } catch {
    // silently fail
  }
}

function formatLogTime(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleTimeString()
}

async function approveGate1() {
  if (selectedIdx.value === null) return
  responding.value = true
  try {
    await respondToGate(props.runId, 'approve', selectedIdx.value)
    selectedIdx.value = null
    await fetchRun()
  } finally {
    responding.value = false
  }
}

function toggleConcept(idx) {
  const i = selectedConceptIndices.value.indexOf(idx)
  if (i >= 0) {
    selectedConceptIndices.value.splice(i, 1)
  } else if (selectedConceptIndices.value.length < 3) {
    selectedConceptIndices.value.push(idx)
  }
}

async function approveConceptSelection() {
  if (selectedConceptIndices.value.length === 0) return
  responding.value = true
  try {
    await respondToGate(props.runId, 'approve', null, selectedConceptIndices.value)
    selectedConceptIndices.value = []
    await fetchRun()
  } finally {
    responding.value = false
  }
}

async function approveGate() {
  responding.value = true
  try {
    await respondToGate(props.runId, 'approve')
    await fetchRun()
  } finally {
    responding.value = false
  }
}

async function rejectGate() {
  responding.value = true
  try {
    await respondToGate(props.runId, 'reject')
    await fetchRun()
  } finally {
    responding.value = false
  }
}

function viewResult() {
  router.push({ name: 'result', params: { filename: run.value.output_path?.split('/').pop()?.replace('.md', '.json') || '' } })
}

onMounted(() => {
  fetchRun()
  fetchLogs()
  pollInterval = setInterval(() => {
    fetchRun()
    fetchLogs()
  }, 2000)
})

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
})
</script>

<style scoped>
.run-header {
  margin-bottom: 1rem;
}

.run-header-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.run-header h2 {
  font-size: 1.5rem;
}

.run-meta {
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
  margin-top: 0.5rem;
}

.pipeline-progress {
  margin-bottom: 1.5rem;
}

.pipeline-progress h3 {
  margin-bottom: 1rem;
  font-size: 1rem;
}

.stages {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.stage-item {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.375rem 0.75rem;
  border-radius: 999px;
  font-size: 0.75rem;
  background: #f1f5f9;
  color: var(--color-text-secondary);
}

.stage-item.active {
  background: #dbeafe;
  color: #1d4ed8;
  font-weight: 600;
}

.stage-item.done {
  background: #dcfce7;
  color: #166534;
}

.stage-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  opacity: 0.5;
}

.stage-item.active .stage-dot {
  opacity: 1;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.4); }
}

.gate-card {
  margin-bottom: 1.5rem;
  border-left: 4px solid var(--color-warning);
}

.tracing-progress-card {
  margin-bottom: 1.5rem;
  border-left: 4px solid var(--color-primary);
}

.tracing-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  margin-bottom: 1rem;
}

.tracing-stat {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.stat-label {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.stat-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--color-text);
}

.progress-bar-container {
  height: 6px;
  background: #e2e8f0;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 0.75rem;
}

.progress-bar {
  height: 100%;
  background: var(--color-primary);
  border-radius: 3px;
  transition: width 0.5s ease;
}

.tracing-current {
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
  font-style: italic;
}

.concept-abstractions {
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid #e2e8f0;
}

.concept-chain {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.25rem;
  margin-top: 0.375rem;
}

.concept-step {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.concept-name {
  padding: 0.125rem 0.5rem;
  border-radius: 999px;
  font-size: 0.75rem;
  background: #f1f5f9;
  color: var(--color-text);
  font-weight: 500;
}

.concept-name.concept-current {
  background: #dbeafe;
  color: #1d4ed8;
  font-weight: 600;
}

.concept-arrow {
  color: var(--color-text-secondary);
  font-size: 0.75rem;
}

.gate-card h3 {
  margin-bottom: 0.5rem;
}

.gate-desc {
  color: var(--color-text-secondary);
  margin-bottom: 1rem;
  font-size: 0.875rem;
}

.candidates {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}

.candidate {
  border: 2px solid var(--color-border);
  border-radius: var(--radius);
  padding: 1rem;
  cursor: pointer;
  transition: all 0.2s;
}

.candidate:hover {
  border-color: var(--color-primary);
}

.candidate.selected {
  border-color: var(--color-primary);
  background: #eff6ff;
}

.candidate-header {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
}

.candidate-num {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--color-primary);
  color: white;
  font-size: 0.75rem;
  font-weight: 700;
}

.candidate-header h4 {
  font-size: 0.9375rem;
  line-height: 1.4;
}

.candidate-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 0.5rem;
  padding-left: 2.25rem;
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
}

.oa-yes {
  color: var(--color-success);
}

.oa-elsevier {
  color: var(--color-success);
}

.oa-no {
  color: var(--color-danger);
}

.doi-link {
  color: var(--color-primary);
  text-decoration: none;
  font-weight: 600;
}

.doi-link:hover {
  text-decoration: underline;
}

.sjr-score {
  font-weight: 600;
  color: #7c3aed;
}

.candidate-abstract {
  margin-top: 0.5rem;
  padding-left: 2.25rem;
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
  line-height: 1.5;
}

.gate-actions {
  display: flex;
  gap: 1rem;
}

.concept-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.concept-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.625rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: all 0.15s;
}

.concept-item:hover {
  border-color: var(--color-primary);
  background: rgba(37, 99, 235, 0.04);
}

.concept-item.selected {
  border-color: var(--color-primary);
  background: rgba(37, 99, 235, 0.08);
}

.concept-check {
  font-size: 1.125rem;
}

.concept-text {
  font-weight: 500;
}

.concept-count {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  margin-bottom: 1rem;
}

.synthesis-content {
  margin-bottom: 1rem;
  background: #f8fafc;
  border-radius: var(--radius);
  padding: 1rem;
  max-height: 400px;
  overflow-y: auto;
}

.synthesis-content h4 {
  margin-bottom: 0.5rem;
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.synthesis-content pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: inherit;
  font-size: 0.875rem;
  line-height: 1.6;
}

.synthesis-content.challenger {
  border-left: 3px solid var(--color-warning);
}

.completed-card {
  border-left: 4px solid var(--color-success);
}

.completed-card h3 {
  color: var(--color-success);
  margin-bottom: 0.5rem;
}

.completed-card code {
  background: #f1f5f9;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.8125rem;
}

.completed-card .btn {
  margin-top: 1rem;
}

.failed-card {
  border-left: 4px solid var(--color-danger);
}

.failed-card h3 {
  color: var(--color-danger);
  margin-bottom: 0.5rem;
}

.error {
  color: var(--color-danger);
}

.loading-card {
  text-align: center;
  padding: 3rem;
  color: var(--color-text-secondary);
}

/* Log Panel */
.log-panel {
  margin-top: 1.5rem;
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.log-header h3 {
  font-size: 1rem;
}

.log-count {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.log-entries {
  background: #1e293b;
  border-radius: var(--radius);
  padding: 0.75rem;
  max-height: 300px;
  overflow-y: auto;
  font-family: 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.75rem;
  line-height: 1.6;
}

.log-empty {
  color: #94a3b8;
  text-align: center;
  padding: 2rem;
}

.log-entry {
  display: flex;
  gap: 0.5rem;
  padding: 0.125rem 0;
  color: #e2e8f0;
}

.log-time {
  color: #64748b;
  flex-shrink: 0;
}

.log-level {
  flex-shrink: 0;
  min-width: 3.5rem;
  font-weight: 600;
}

.log-source {
  color: #7dd3fc;
  flex-shrink: 0;
  min-width: 8rem;
}

.log-msg {
  word-break: break-word;
}

.log-entry.log-info .log-level {
  color: #22d3ee;
}

.log-entry.log-warning .log-level {
  color: #fbbf24;
}

.log-entry.log-error {
  background: rgba(239, 68, 68, 0.15);
  border-radius: 2px;
}

.log-entry.log-error .log-level {
  color: #f87171;
}

.log-entry.log-debug .log-level {
  color: #a78bfa;
}

.graph-loading {
  color: var(--color-text-secondary);
  text-align: center;
  padding: 2rem;
  font-style: italic;
}
</style>
