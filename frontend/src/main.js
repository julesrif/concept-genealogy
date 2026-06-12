import { createApp } from 'vue'
import { createPinia } from 'pinia'
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
app.use(createPinia())
app.use(router)
app.mount('#app')
