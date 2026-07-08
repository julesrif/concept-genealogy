/**
 * api.js - REST client for the Concept Genealogy backend.
 *
 * Design decision: thin wrapper using native fetch() rather than axios/ky because
 * the API surface is small (6 endpoints) and all responses are JSON. Adding a
 * library dependency for fetch ergonomics is not justified at this scale.
 *
 * The API_BASE uses a relative path ('/api') which is proxied to the backend by
 * Vite's dev server (vite.config.js proxy) in development and by nginx in
 * production (nginx.conf reverse proxy). This avoids CORS configuration entirely.
 */
const API_BASE = '/api'

export async function startRun(concept, { depth = null, papersPerLevel = null, doi = null } = {}) {
  const body = {}
  if (doi) {
    body.doi = doi
  } else {
    body.concept = concept
  }
  if (depth !== null) body.depth = depth
  if (papersPerLevel !== null) body.papers_per_level = papersPerLevel

  const res = await fetch(`${API_BASE}/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Failed to start run: ${res.statusText}`)
  return res.json()
}

export async function getRun(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}`)
  if (!res.ok) throw new Error(`Failed to get run: ${res.statusText}`)
  return res.json()
}

export async function listRuns() {
  const res = await fetch(`${API_BASE}/runs`)
  if (!res.ok) throw new Error(`Failed to list runs: ${res.statusText}`)
  return res.json()
}

export async function respondToGate(runId, action, selectedIndex = null, selectedIndices = null) {
  const body = { action }
  if (selectedIndex !== null) body.selected_index = selectedIndex
  if (selectedIndices !== null) body.selected_indices = selectedIndices

  const res = await fetch(`${API_BASE}/runs/${runId}/gate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Failed to respond to gate: ${res.statusText}`)
  return res.json()
}

export async function getLogs(runId, since = 0) {
  const res = await fetch(`${API_BASE}/runs/${runId}/logs?since=${since}`)
  if (!res.ok) throw new Error(`Failed to get logs: ${res.statusText}`)
  return res.json()
}

export async function getResult(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}/result`)
  if (!res.ok) throw new Error(`Failed to get result: ${res.statusText}`)
  return res.json()
}

export async function listOutputs() {
  const res = await fetch(`${API_BASE}/outputs`)
  if (!res.ok) throw new Error(`Failed to list outputs: ${res.statusText}`)
  return res.json()
}

export async function getOutput(filename) {
  const res = await fetch(`${API_BASE}/outputs/${filename}`)
  if (!res.ok) throw new Error(`Failed to get output: ${res.statusText}`)
  return res.json()
}

export async function getGraph(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}/graph`)
  if (!res.ok) throw new Error(`Failed to get graph: ${res.statusText}`)
  return res.json()
}
