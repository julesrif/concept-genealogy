<template>
  <div class="home">
    <div class="hero">
      <h2>Academic Concept Tracing</h2>
      <p>
        Enter an academic concept to trace its genealogy through citation chains.
        The system will identify a seed paper, recursively extract claims, and
        build a semantic provenance graph.
      </p>
    </div>

    <form class="card run-form" @submit.prevent="handleSubmit">
      <div class="form-group">
        <label for="concept">Research Concept</label>
        <input
          id="concept"
          v-model="concept"
          type="text"
          placeholder="e.g. attention mechanism, reinforcement learning from human feedback"
          required
          :disabled="loading"
        />
      </div>
      <div class="form-group">
        <label for="depth">Max Citation Depth (optional)</label>
        <input
          id="depth"
          v-model.number="depth"
          type="number"
          min="1"
          max="5"
          placeholder="Default: 3"
          :disabled="loading"
        />
      </div>
      <div class="form-group">
        <label for="papers-per-level">Papers per Level (optional)</label>
        <input
          id="papers-per-level"
          v-model.number="papersPerLevel"
          type="number"
          min="1"
          max="10"
          placeholder="Default: 3"
          :disabled="loading"
        />
      </div>
      <button type="submit" class="btn btn-primary" :disabled="loading || !concept.trim()">
        {{ loading ? 'Starting...' : 'Start Tracing' }}
      </button>
      <p v-if="error" class="error">{{ error }}</p>
    </form>

    <div v-if="recentRuns.length" class="recent-section">
      <h3>Active Runs</h3>
      <div class="runs-list">
        <router-link
          v-for="run in recentRuns"
          :key="run.run_id"
          :to="{ name: 'run', params: { runId: run.run_id } }"
          class="run-item card"
        >
          <div class="run-item-header">
            <strong>{{ run.user_concept }}</strong>
            <span :class="'badge badge-' + statusClass(run.status)">{{ run.status }}</span>
          </div>
          <div class="run-item-meta">
            <span>Stage: {{ run.stage }}</span>
            <span>{{ formatDate(run.created_at) }}</span>
          </div>
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { startRun, listRuns } from '../api.js'

const router = useRouter()
const concept = ref('')
const depth = ref(null)
const papersPerLevel = ref(null)
const loading = ref(false)
const error = ref('')
const recentRuns = ref([])
let pollInterval = null

async function handleSubmit() {
  loading.value = true
  error.value = ''
  try {
    const run = await startRun(concept.value.trim(), {
      depth: depth.value || null,
      papersPerLevel: papersPerLevel.value || null,
    })
    router.push({ name: 'run', params: { runId: run.run_id } })
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function fetchRuns() {
  try {
    const runs = await listRuns()
    recentRuns.value = runs.filter(
      r => r.status === 'running' || r.status === 'awaiting_input'
    )
  } catch {
    // silently fail on polling
  }
}

function statusClass(status) {
  const map = { running: 'running', awaiting_input: 'awaiting', completed: 'completed', failed: 'failed' }
  return map[status] || 'running'
}

function formatDate(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString()
}

onMounted(() => {
  fetchRuns()
  pollInterval = setInterval(fetchRuns, 5000)
})

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
})
</script>

<style scoped>
.hero {
  text-align: center;
  margin-bottom: 2rem;
}

.hero h2 {
  font-size: 1.75rem;
  margin-bottom: 0.5rem;
}

.hero p {
  color: var(--color-text-secondary);
  max-width: 600px;
  margin: 0 auto;
}

.run-form {
  max-width: 600px;
  margin: 0 auto 2rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  font-weight: 600;
  margin-bottom: 0.375rem;
  font-size: 0.875rem;
}

.form-group input {
  width: 100%;
  padding: 0.625rem 0.875rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  font-size: 0.9375rem;
  transition: border-color 0.2s;
}

.form-group input:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.error {
  color: var(--color-danger);
  margin-top: 0.5rem;
  font-size: 0.875rem;
}

.recent-section {
  max-width: 600px;
  margin: 0 auto;
}

.recent-section h3 {
  margin-bottom: 1rem;
}

.runs-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.run-item {
  text-decoration: none;
  color: inherit;
  transition: box-shadow 0.2s;
}

.run-item:hover {
  box-shadow: var(--shadow-lg);
}

.run-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.run-item-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
}
</style>
