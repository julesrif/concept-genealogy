<template>
  <div class="history">
    <h2>Run History</h2>
    <p class="subtitle">Previously completed pipeline outputs</p>

    <div v-if="loading" class="card loading-card">
      <p>Loading...</p>
    </div>

    <div v-else-if="outputs.length === 0" class="card empty-card">
      <p>No completed runs yet. Start a new run from the home page.</p>
    </div>

    <div v-else class="outputs-list">
      <div
        v-for="output in outputs"
        :key="output.filename"
        class="card output-item"
        @click="viewOutput(output.filename)"
      >
        <div class="output-header">
          <h4>{{ output.user_concept }}</h4>
          <span class="badge badge-completed">completed</span>
        </div>
        <div class="output-meta">
          <span v-if="output.seed_paper_title">Seed: {{ output.seed_paper_title }}</span>
          <span>{{ output.filename }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { listOutputs } from '../api.js'

const router = useRouter()
const outputs = ref([])
const loading = ref(true)

async function fetchOutputs() {
  try {
    outputs.value = await listOutputs()
  } catch {
    // ignore
  } finally {
    loading.value = false
  }
}

function viewOutput(filename) {
  router.push({ name: 'result', params: { filename } })
}

onMounted(fetchOutputs)
</script>

<style scoped>
.history h2 {
  margin-bottom: 0.25rem;
}

.subtitle {
  color: var(--color-text-secondary);
  margin-bottom: 1.5rem;
}

.outputs-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.output-item {
  cursor: pointer;
  transition: box-shadow 0.2s;
}

.output-item:hover {
  box-shadow: var(--shadow-lg);
}

.output-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.output-header h4 {
  text-transform: capitalize;
}

.output-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.8125rem;
  color: var(--color-text-secondary);
}

.loading-card,
.empty-card {
  text-align: center;
  padding: 3rem;
  color: var(--color-text-secondary);
}
</style>
