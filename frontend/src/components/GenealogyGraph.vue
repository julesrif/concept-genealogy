<!--
  GenealogyGraph.vue - Interactive citation DAG visualisation.

  Design decisions:
  - v-network-graph chosen over D3.js because it provides a declarative, Vue-native
    graph renderer with built-in pan/zoom, node dragging, and edge labels - avoids
    the imperative DOM manipulation that D3 requires and integrates cleanly with
    Vue's reactivity system (computed props trigger re-renders automatically).
  - dagre layout algorithm (Sugiyama-style hierarchical) is used for automatic node
    positioning because the citation graph is a directed acyclic graph where depth
    (generation) should map to vertical layers - dagre optimises for minimal edge
    crossings in layered digraphs, producing readable academic genealogy trees.
  - Depth-based colour coding provides immediate visual semantics: seed paper (amber),
    depth-1 (blue), depth-2 (purple), etc. - users can identify citation generations
    at a glance without reading labels.
  - The component is stateless (props-only) and reusable - it receives graphData and
    renders it, with no knowledge of the pipeline or API. This follows the Presentational
    Component pattern (separation of data-fetching from rendering).
-->
<script setup>
import { computed, ref } from 'vue'
import { VNetworkGraph, VEdgeLabel } from 'v-network-graph'
import 'v-network-graph/lib/style.css'
import dagre from 'dagre'

const props = defineProps({
  graphData: { type: Object, default: () => ({ nodes: [], edges: [], concept_at_depth: {} }) },
})

// Depth-based colours
const DEPTH_COLORS = ['#f59e0b', '#3b82f6', '#8b5cf6', '#10b981', '#ef4444', '#6366f1']

function colorForDepth(depth) {
  if (depth === 0) return DEPTH_COLORS[0]
  return DEPTH_COLORS[Math.min(depth, DEPTH_COLORS.length - 1)]
}

// Build v-network-graph data structures from API response
const nodes = computed(() => {
  const result = {}
  for (const n of props.graphData.nodes || []) {
    const authorStr = (n.authors || []).slice(0, 2).join(', ')
    result[n.id] = {
      name: n.label,
      authors: authorStr,
      doi: n.doi || '',
      type: n.type,
      depth: n.depth ?? 0,
    }
  }
  return result
})

const edges = computed(() => {
  const result = {}
  const edgeList = props.graphData.edges || []
  for (let i = 0; i < edgeList.length; i++) {
    const e = edgeList[i]
    result[`edge${i}`] = { source: e.source, target: e.target, concept: e.concept || '' }
  }
  return result
})

// Compute dagre layout (top-to-bottom)
const layouts = computed(() => {
  const nodeList = props.graphData.nodes || []
  const edgeList = props.graphData.edges || []
  if (nodeList.length === 0) return { nodes: {} }

  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'TB', ranksep: 150, nodesep: 100 })
  g.setDefaultEdgeLabel(() => ({}))

  for (const n of nodeList) {
    g.setNode(n.id, { width: 220, height: 72 })
  }

  for (const e of edgeList) {
    g.setEdge(e.source, e.target)
  }

  dagre.layout(g)

  const positions = {}
  for (const n of nodeList) {
    const pos = g.node(n.id)
    if (pos) {
      positions[n.id] = { x: pos.x, y: pos.y }
    }
  }
  return { nodes: positions }
})

// Max depth for legend
const maxDepth = computed(() => {
  let max = 0
  for (const n of props.graphData.nodes || []) {
    if ((n.depth ?? 0) > max) max = n.depth
  }
  return max
})

// ── Path highlighting & breadcrumbs ────────────────────────────────────────
const selectedNodeId = ref(null)

// Find seed node ID
const seedNodeId = computed(() => {
  const seed = (props.graphData.nodes || []).find((n) => n.type === 'seed')
  return seed ? seed.id : null
})

// BFS from seed to target, returns array of node IDs on the path
function findPathToSeed(targetId) {
  // Build adjacency - only use edges going from shallower to deeper depth
  // (true parent→child edges). Cross-depth or same-depth references can form
  // cycles which break the ancestor walk.
  const parentMap = {}
  const edgeByPair = {}
  for (const [edgeId, e] of Object.entries(edges.value)) {
    const sourceDepth = nodes.value[e.source]?.depth ?? 0
    const targetDepth = nodes.value[e.target]?.depth ?? 0
    if (sourceDepth < targetDepth) {
      // Prefer the edge from the shallowest parent (closest to seed)
      if (!(e.target in parentMap) || sourceDepth < (nodes.value[parentMap[e.target]]?.depth ?? 0)) {
        parentMap[e.target] = e.source
        edgeByPair[`${e.source}->${e.target}`] = edgeId
      }
    }
  }

  // Walk from target back to seed following parent links
  const path = []
  let current = targetId
  const visited = new Set()
  while (current && current !== seedNodeId.value && !visited.has(current)) {
    visited.add(current)
    const parent = parentMap[current]
    if (!parent) break
    path.unshift({ from: parent, to: current, edgeId: edgeByPair[`${parent}->${current}`] })
    current = parent
  }

  if (current === seedNodeId.value) return path
  return [] // no path found
}

// Highlighted path (list of edge IDs)
const highlightedPath = computed(() => {
  if (!selectedNodeId.value || selectedNodeId.value === seedNodeId.value) return {}
  const pathEdges = findPathToSeed(selectedNodeId.value)
  if (pathEdges.length === 0) return {}
  return {
    highlight: {
      edges: pathEdges.map((p) => p.edgeId).filter(Boolean),
      color: '#ef4444',
    },
  }
})

// Breadcrumbs: ordered list of { nodeId, nodeName, concept (on the incoming edge) }
const breadcrumbs = computed(() => {
  if (!selectedNodeId.value || selectedNodeId.value === seedNodeId.value) return []
  const pathEdges = findPathToSeed(selectedNodeId.value)
  if (pathEdges.length === 0) return []

  const crumbs = []
  // First crumb: seed
  const seedNode = nodes.value[seedNodeId.value]
  crumbs.push({ nodeId: seedNodeId.value, name: seedNode?.name || 'Seed', concept: null })

  // Subsequent crumbs: each node along the path with incoming edge concept
  for (const step of pathEdges) {
    const edgeData = edges.value[step.edgeId]
    const nodeData = nodes.value[step.to]
    crumbs.push({
      nodeId: step.to,
      name: nodeData?.name || step.to,
      concept: edgeData?.concept || null,
    })
  }
  return crumbs
})

// Event handlers for node click
const eventHandlers = {
  'node:click': ({ node }) => {
    if (selectedNodeId.value === node) {
      selectedNodeId.value = null // toggle off
    } else {
      selectedNodeId.value = node
    }
  },
  'view:click': () => {
    selectedNodeId.value = null // deselect on background click
  },
}

// Build node label as multiline string: Title\nAuthor\nDOI
function nodeLabel(node) {
  const lines = [node.name || '']
  if (node.authors) lines.push(node.authors)
  if (node.doi) lines.push(`DOI: ${node.doi}`)
  return lines.join('\n')
}

// Configs for node appearance
const configs = computed(() => ({
  view: {
    scalingObjects: true,
    minZoomLevel: 0.2,
    maxZoomLevel: 3,
  },
  node: {
    normal: {
      radius: (node) => node.type === 'seed' ? 28 : 20,
      color: (node) => colorForDepth(node.depth ?? 0),
    },
    label: {
      fontSize: 9,
      color: '#1e293b',
      direction: 'south',
      text: nodeLabel,
    },
  },
  edge: {
    normal: {
      color: '#94a3b8',
      width: 1.5,
    },
    marker: {
      target: { type: 'arrow', width: 4, height: 4 },
    },
    label: {
      fontSize: 10,
      color: '#475569',
      background: {
        visible: true,
        color: '#f1f5f9',
        padding: { vertical: 1, horizontal: 4 },
        borderRadius: 3,
      },
    },
  },
  path: {
    visible: true,
    normal: {
      width: 6,
      dasharray: '',
      animate: false,
      animationSpeed: 0,
    },
  },
}))
</script>

<template>
  <div class="genealogy-graph-wrapper">
    <div class="graph-legend">
      <span v-for="d in (maxDepth + 1)" :key="d - 1" class="legend-item">
        <span class="legend-dot" :style="{ background: colorForDepth(d - 1) }"></span>
        {{ d - 1 === 0 ? 'Seed (depth 0)' : `Depth ${d - 1}` }}
      </span>
      <span class="legend-flow">Click a node to trace concept path back to seed</span>
    </div>
    <v-network-graph
      class="genealogy-graph"
      :nodes="nodes"
      :edges="edges"
      :paths="highlightedPath"
      :layouts="layouts"
      :configs="configs"
      :event-handlers="eventHandlers"
    >
      <template #edge-label="{ edge, ...slotProps }">
        <v-edge-label :text="edge.concept" align="center" vertical-align="above" v-bind="slotProps" />
      </template>
    </v-network-graph>

    <!-- Breadcrumbs: concept trail from seed to selected node -->
    <div v-if="breadcrumbs.length > 0" class="concept-breadcrumbs">
      <span class="breadcrumbs-label">Concept trail:</span>
      <template v-for="(crumb, idx) in breadcrumbs" :key="crumb.nodeId">
        <span v-if="crumb.concept" class="breadcrumb-concept">{{ crumb.concept }}</span>
        <span v-if="crumb.concept" class="breadcrumb-arrow">→</span>
        <span
          class="breadcrumb-node"
          :class="{ 'breadcrumb-seed': idx === 0, 'breadcrumb-selected': idx === breadcrumbs.length - 1 }"
        >{{ crumb.name }}</span>
        <span v-if="idx < breadcrumbs.length - 1" class="breadcrumb-arrow">→</span>
      </template>
    </div>
  </div>
</template>

<style scoped>
.genealogy-graph-wrapper {
  width: 100%;
}

.genealogy-graph {
  width: 100%;
  height: 500px;
  border: 1px solid var(--color-border, #e2e8f0);
  border-radius: var(--radius, 8px);
  background: #f8fafc;
}

/* Bold node labels */
.genealogy-graph :deep(.v-ng-node-labels text) {
  font-weight: 700;
}

.graph-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem;
  margin-bottom: 0.75rem;
  font-size: 0.8125rem;
  color: var(--color-text-secondary, #64748b);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.legend-flow {
  margin-left: auto;
  font-style: italic;
  color: #94a3b8;
}

/* Breadcrumbs */
.concept-breadcrumbs {
  margin-top: 0.75rem;
  padding: 0.75rem 1rem;
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.8125rem;
  line-height: 1.5;
}

.breadcrumbs-label {
  font-weight: 600;
  color: #334155;
  margin-right: 0.25rem;
}

.breadcrumb-node {
  padding: 0.125rem 0.5rem;
  border-radius: 4px;
  background: #e2e8f0;
  color: #1e293b;
  font-weight: 500;
}

.breadcrumb-seed {
  background: #fef3c7;
  color: #92400e;
  font-weight: 600;
}

.breadcrumb-selected {
  background: #dbeafe;
  color: #1e40af;
  font-weight: 600;
}

.breadcrumb-concept {
  padding: 0.125rem 0.5rem;
  border-radius: 12px;
  background: #ede9fe;
  color: #6d28d9;
  font-style: italic;
  font-size: 0.75rem;
}

.breadcrumb-arrow {
  color: #94a3b8;
  font-size: 0.75rem;
}
</style>
