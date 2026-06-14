<script setup>
import { ref, watch, computed } from 'vue'
import { VNetworkGraph } from 'v-network-graph'
import 'v-network-graph/lib/style.css'
import dagre from 'dagre'

const props = defineProps({
  graphData: { type: Object, default: () => ({ nodes: [], edges: [], concepts: [] }) },
})

// Build v-network-graph data structures from API response
const nodes = computed(() => {
  const result = {}
  for (const n of props.graphData.nodes || []) {
    result[n.id] = { name: n.label, type: n.type, depth: n.depth }
  }
  return result
})

const edges = computed(() => {
  const result = {}
  const edgeList = props.graphData.edges || []
  for (let i = 0; i < edgeList.length; i++) {
    const e = edgeList[i]
    result[`edge${i}`] = { source: e.source, target: e.target }
  }
  return result
})

// Compute dagre layout (top-to-bottom)
const layouts = computed(() => {
  const nodeList = props.graphData.nodes || []
  const edgeList = props.graphData.edges || []
  if (nodeList.length === 0) return { nodes: {} }

  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'TB', ranksep: 100, nodesep: 60 })
  g.setDefaultEdgeLabel(() => ({}))

  for (const n of nodeList) {
    const width = n.type === 'concept' ? 120 : 180
    const height = n.type === 'concept' ? 36 : 48
    g.setNode(n.id, { width, height })
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

// Configs for node appearance
const configs = computed(() => ({
  view: {
    scalingObjects: true,
    minZoomLevel: 0.3,
    maxZoomLevel: 2,
  },
  node: {
    normal: {
      radius: (node) => node.type === 'concept' ? 16 : 24,
      color: (node) => {
        if (node.type === 'concept') return '#8b5cf6'
        if (node.depth === 0) return '#f59e0b'
        return '#3b82f6'
      },
    },
    label: {
      fontSize: 11,
      color: '#1e293b',
      direction: 'south',
      text: 'name',
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
  },
}))
</script>

<template>
  <div class="genealogy-graph-wrapper">
    <div class="graph-legend">
      <span class="legend-item"><span class="legend-dot seed"></span> Seed Paper</span>
      <span class="legend-item"><span class="legend-dot concept"></span> Concept</span>
      <span class="legend-item"><span class="legend-dot reference"></span> Reference Paper</span>
      <span class="legend-flow">Flow: Seed -> Concept -> References</span>
    </div>
    <v-network-graph
      class="genealogy-graph"
      :nodes="nodes"
      :edges="edges"
      :layouts="layouts"
      :configs="configs"
    />
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

.graph-legend {
  display: flex;
  gap: 1.5rem;
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

.legend-dot.seed {
  background: #f59e0b;
}

.legend-dot.concept {
  background: #8b5cf6;
}

.legend-dot.reference {
  background: #3b82f6;
}

.legend-flow {
  margin-left: auto;
  font-style: italic;
  color: #94a3b8;
}
</style>
