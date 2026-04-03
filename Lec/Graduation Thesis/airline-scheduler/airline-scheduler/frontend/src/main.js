import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import SchedulerView from './views/SchedulerView.vue'

const router = createRouter({
  history: createWebHistory('/'),
  routes: [{ path: '/', component: SchedulerView }]
})

createApp(App).use(router).mount('#app')
