/**
 * main.js - Application entry point.
 *
 * Design decision: vue-router uses createWebHistory (HTML5 history mode) rather
 * than hash mode because the nginx production config (nginx.conf) includes a
 * try_files fallback to index.html, enabling clean URLs (/run/abc vs /#/run/abc).
 * This is standard for SPAs served behind a reverse proxy.
 *
 * Route structure maps 1:1 to the pipeline lifecycle:
 *   /            → HomeView (launch a run)
 *   /run/:runId  → RunView (monitor active pipeline, interact with HITL gates)
 *   /history     → HistoryView (browse completed runs)
 *   /result/:id  → ResultView (view final output: narrative, graph, claims)
 */
import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import HomeView from './views/HomeView.vue'
import RunView from './views/RunView.vue'
import HistoryView from './views/HistoryView.vue'
import ResultView from './views/ResultView.vue'

const routes = [
  { path: '/', name: 'home', component: HomeView },
  { path: '/run/:runId', name: 'run', component: RunView, props: true },
  { path: '/history', name: 'history', component: HistoryView },
  { path: '/result/:filename', name: 'result', component: ResultView, props: true },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

const app = createApp(App)
app.use(router)
app.mount('#app')
