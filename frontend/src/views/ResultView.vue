<!--
  ResultView.vue — Completed run display (narrative, graph, claims table).

  Design decisions:
  - Separates the completed result view from the live RunView because the data shape
    differs: ResultView reads a static JSON output file whereas RunView polls mutable
    pipeline state. Separation follows the Single Responsibility Principle.
  - The GenealogyGraph component is embedded here (not in RunView) because the graph
    data is only available after the pipeline completes — rendering mid-pipeline would
    show an incomplete/misleading visualisation.
  - Claims are displayed in a sortable table with depth and confidence columns so the
    user can assess synthesis quality — supports the critical evaluation goal of the
    HITL design (Mosqueira-Rey et al., 2023).
  - Narrative is rendered with v-html from the backend Markdown-to-HTML conversion,
    as the synthesiser produces structured Markdown with headers and citations.
-->
<template>
  <div class="result-view">
    <div v-if="loading" class="card loading-card">
      <p>Loading result...</p>
    </div>

    <div v-else-if="error" class="card error-card">
      <p class="error">{{ error }}</p>
      <router-link to="/history" class="btn btn-primary">Back to History</router-link>
    </div>

    <template v-else-if="result">
      <!-- Header -->
      <div class="result-header card">
        <h2>{{ result.user_concept }}</h2>
        <p class="run-meta">Run: {{ result.run_id }}</p>
      </div>

      <!-- Seed Paper -->
      <div v-if="result.seed_paper" class="card section">
        <h3>🌱 Seed Paper</h3>
        <div class="seed-info">
          <h4>{{ result.seed_paper.title }}</h4>
          <p class="meta">
            {{ (result.seed_paper.authors || []).join(', ') }}
            ({{ result.seed_paper.year || 'N/A' }})
          </p>
          <p class="meta">Citations: {{ result.seed_paper.citation_count }}</p>
          <p v-if="result.seed_paper.abstract" class="abstract">
            {{ result.seed_paper.abstract }}
          </p>
        </div>
      </div>

      <!-- Tracing Results -->
      <div v-if="result.tracing_results?.length" class="card section">
        <h3>🔗 Citation Trace ({{ result.tracing_results.length }} levels)</h3>
        <div
          v-for="(trace, idx) in result.tracing_results"
          :key="idx"
          class="trace-level"
        >
          <div class="trace-header">
            <span class="depth-badge">Depth {{ trace.depth }}</span>
            <strong>{{ trace.paper_title }}</strong>
          </div>
          <div class="claims-list">
            <div v-for="(claim, cIdx) in trace.claims" :key="cIdx" class="claim-item">
              <p class="claim-text">"{{ claim.claim_text }}"</p>
              <div class="claim-meta">
                <span>Validity: {{ (claim.validity_score * 100).toFixed(0) }}%</span>
                <span v-if="claim.drift_score != null">
                  Drift: {{ (claim.drift_score * 100).toFixed(0) }}%
                </span>
                <span v-if="claim.relevant_concepts?.length">
                  Concepts: {{ claim.relevant_concepts.join(', ') }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Synthesis -->
      <div v-if="result.synthesis" class="card section">
        <h3>📝 Synthesis</h3>
        <div class="synthesis-text">
          <pre>{{ result.synthesis }}</pre>
        </div>
      </div>

      <!-- Challenger Critique -->
      <div v-if="result.challenger_critique" class="card section challenger-section">
        <h3>⚔️ Challenger Critique</h3>
        <div class="synthesis-text">
          <pre>{{ result.challenger_critique }}</pre>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getOutput } from '../api.js'

const props = defineProps({ filename: String })

const result = ref(null)
const loading = ref(true)
const error = ref('')

onMounted(async () => {
  try {
    result.value = await getOutput(props.filename)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.result-header {
  margin-bottom: 1.5rem;
}

.result-header h2 {
  text-transform: capitalize;
}

.run-meta {
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
  margin-top: 0.25rem;
}

.section {
  margin-bottom: 1.5rem;
}

.section h3 {
  margin-bottom: 1rem;
  font-size: 1.125rem;
}

.seed-info h4 {
  margin-bottom: 0.5rem;
}

.seed-info .meta {
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
  margin-bottom: 0.25rem;
}

.seed-info .abstract {
  margin-top: 0.75rem;
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--color-text-secondary);
}

.trace-level {
  margin-bottom: 1.5rem;
  padding-left: 1rem;
  border-left: 3px solid var(--color-primary);
}

.trace-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}

.depth-badge {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  background: var(--color-primary);
  color: white;
  border-radius: 999px;
  font-size: 0.6875rem;
  font-weight: 700;
  text-transform: uppercase;
}

.claims-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.claim-item {
  background: #f8fafc;
  padding: 0.75rem 1rem;
  border-radius: var(--radius);
}

.claim-text {
  font-size: 0.875rem;
  font-style: italic;
  line-height: 1.5;
  margin-bottom: 0.5rem;
}

.claim-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
}

.synthesis-text {
  background: #f8fafc;
  padding: 1rem;
  border-radius: var(--radius);
  max-height: 500px;
  overflow-y: auto;
}

.synthesis-text pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: inherit;
  font-size: 0.875rem;
  line-height: 1.6;
}

.challenger-section {
  border-left: 4px solid var(--color-warning);
}

.loading-card,
.error-card {
  text-align: center;
  padding: 3rem;
}

.error {
  color: var(--color-danger);
  margin-bottom: 1rem;
}
</style>
