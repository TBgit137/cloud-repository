<template>
  <div class="page">

    <header class="header">
      <div class="header__inner">
        <div class="header__brand">
          <div class="header__logo">
            <img src="/plane.png" alt="logo" class="header__logo-img" />
          </div>
          <span class="header__title">Airline Scheduler</span>
        </div>
        <span class="header__badge">Beta</span>
      </div>
    </header>

    <main class="main">
      <div class="hero">
        <h1 class="hero__heading">航班跑道智能调度</h1>
        <p class="hero__sub">上传航班数据集，配置调度参数，获取优化后的跑道排班方案</p>
      </div>

      <div class="card-grid">
        <section class="card">
          <div class="card__head">
            <span class="card__num">01</span>
            <h2 class="card__title">上传数据集</h2>
          </div>
          <FileDropZone v-model:file="form.file" />
        </section>

        <section class="card">
          <div class="card__head">
            <span class="card__num">02</span>
            <h2 class="card__title">调度参数</h2>
          </div>
          <div class="params">
            <div class="param-row">
              <label class="param-label">目标机场代码</label>
              <input v-model="form.airport" class="param-input" type="text"
                placeholder="例：SBGR（默认 SBGR）" maxlength="10" />
            </div>
            <div class="param-row">
              <label class="param-label">可同时使用跑道数</label>
              <input v-model.number="form.nRunways" class="param-input" type="number"
                min="1" max="20" placeholder="默认 3" />
            </div>
            <div class="param-row">
              <label class="param-label">同跑道最短安全间隔（分钟）</label>
              <input v-model.number="form.safetyInterval" class="param-input" type="number"
                min="1" max="60" placeholder="默认 3" />
            </div>
            <div class="param-row">
              <label class="param-label">调度时间范围</label>
              <DateRangePicker v-model="form.dateRange" />
            </div>
          </div>
        </section>
      </div>

      <!-- 提交按钮 -->
      <div class="submit-row">
        <button class="btn-submit" :disabled="!form.file || isRunning" @click="onSubmit">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
          {{ isRunning ? '处理中…' : '开始调度' }}
        </button>
        <p v-if="!form.file && !isRunning && steps.length === 0" class="submit-hint">请先上传 CSV 数据集</p>
      </div>

      <!-- 进度步骤列表 -->
      <div v-if="steps.length > 0" class="progress-box">
        <div v-for="(s, i) in steps" :key="i" class="progress-step"
          :class="{
            'progress-step--active': i === steps.length - 1 && isRunning && !s.isError,
            'progress-step--done':   !isRunning && !s.isError && finalState === 'done' || (i < steps.length - 1 && !s.isError),
            'progress-step--error':  s.isError,
          }">
          <span class="step-icon">
            <template v-if="s.isError">✕</template>
            <template v-else-if="i < steps.length - 1 || (!isRunning && finalState === 'done')">✓</template>
            <template v-else><span class="spinner"></span></template>
          </span>
          <span class="step-msg">{{ s.message }}</span>
        </div>
      </div>

      <!-- 完成摘要 -->
      <div v-if="finalState === 'done' && summary" class="summary-box">
        <span>机场：{{ summary.airport }}</span>
        <span>跑道数：{{ summary.n_runways }}</span>
        <span>总事件：{{ summary.total_events }}</span>
        <span>起飞：{{ summary.departure_events }}</span>
        <span>降落：{{ summary.arrival_events }}</span>
        <span>惩罚值：{{ summary.penalty }}</span>
        <span>原始行数：{{ summary.original_rows }}</span>
            <span>时段内行数：{{ summary.filtered_rows }}</span>
            <span>清洗后：{{ summary.cleaned_rows }}</span>
            <span>时段内清洗掉：{{ summary.removed_rows }}</span>
      </div>

      <template v-if="finalState === 'done'">

        <!-- 排班结果表 -->
        <section class="result-section">
          <div class="result-head">
            <h2 class="result-title">排班结果</h2>
            <span class="result-count">共 {{ schedule.length }} 条</span>
            <div class="result-filters">
              <input v-model="scheduleSearch" class="filter-input" placeholder="搜索航班号 / 跑道…" />
              <select v-model="scheduleOpFilter" class="filter-select">
                <option value="">全部类型</option>
                <option value="departure">起飞</option>
                <option value="arrival">降落</option>
              </select>
            </div>
          </div>
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th @click="sortBy('flight_id')" class="sortable">航班号 <SortIcon col="flight_id" :current="sort" /></th>
                  <th @click="sortBy('operation')" class="sortable">类型 <SortIcon col="operation" :current="sort" /></th>
                  <th @click="sortBy('planned_time')" class="sortable">计划时间 <SortIcon col="planned_time" :current="sort" /></th>
                  <th @click="sortBy('scheduled_time')" class="sortable">排班时间 <SortIcon col="scheduled_time" :current="sort" /></th>
                  <th @click="sortBy('delay_minutes')" class="sortable">延误（分钟）<SortIcon col="delay_minutes" :current="sort" /></th>
                  <th @click="sortBy('runway')" class="sortable">跑道 <SortIcon col="runway" :current="sort" /></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in pagedSchedule" :key="row.flight_id + row.planned_time">
                  <td>{{ row.flight_id }}</td>
                  <td><span class="badge" :class="row.operation === 'departure' ? 'badge--dep' : 'badge--arr'">{{ row.operation === 'departure' ? '起飞' : '降落' }}</span></td>
                  <td class="mono">{{ fmtTime(row.planned_time) }}</td>
                  <td class="mono">{{ fmtTime(row.scheduled_time) }}</td>
                  <td><span class="delay" :class="delayClass(row.delay_minutes)">{{ row.delay_minutes > 0 ? '+' : '' }}{{ row.delay_minutes }}</span></td>
                  <td><span class="runway-badge">R{{ row.runway }}</span></td>
                </tr>
                <tr v-if="filteredSchedule.length === 0">
                  <td colspan="6" class="empty-row">无匹配记录</td>
                </tr>
              </tbody>
            </table>
          </div>
          <Pagination :total="filteredSchedule.length" :page="schedulePage" :pageSize="PAGE_SIZE" @change="p => schedulePage = p" />
        </section>

        <!-- 已清洗航班表 -->
        <section class="result-section" v-if="removedFlights.length > 0">
          <div class="result-head">
            <h2 class="result-title">已清洗航班</h2>
            <span class="result-count removed-count">共 {{ removedFlights.length }} 条</span>
            <div class="result-filters">
              <input v-model="removedSearch" class="filter-input" placeholder="搜索航班号 / 机场…" />
            </div>
          </div>
          <div class="table-wrap">
            <table class="data-table">
              <thead>
                <tr>
                  <th>航班号</th><th>出发机场</th><th>到达机场</th><th>计划起飞</th><th>计划降落</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, i) in pagedRemoved" :key="i">
                  <td>{{ row.flight_no }}</td>
                  <td>{{ row.airport_from }}</td>
                  <td>{{ row.airport_to }}</td>
                  <td class="mono">{{ row.scheduled_departure }}</td>
                  <td class="mono">{{ row.scheduled_arrival }}</td>
                </tr>
                <tr v-if="filteredRemoved.length === 0">
                  <td colspan="5" class="empty-row">无匹配记录</td>
                </tr>
              </tbody>
            </table>
          </div>
          <Pagination :total="filteredRemoved.length" :page="removedPage" :pageSize="PAGE_SIZE" @change="p => removedPage = p" />
        </section>

      </template>
    </main>

    <footer class="footer">Airline Scheduler &copy; 2026</footer>
  </div>
</template>

<script setup>
import { reactive, ref, computed, defineComponent, h } from 'vue'
import FileDropZone from '../components/FileDropZone.vue'
import DateRangePicker from '../components/DateRangePicker.vue'
import { uploadDataset } from '../api/index.js'

const SortIcon = defineComponent({
  props: { col: String, current: Object },
  setup(props) {
    return () => {
      const active = props.current.col === props.col
      return h('span', { class: 'sort-icon' }, active ? (props.current.asc ? ' ▲' : ' ▼') : ' ⇅')
    }
  }
})

const Pagination = defineComponent({
  props: { total: Number, page: Number, pageSize: Number },
  emits: ['change'],
  setup(props, { emit }) {
    const totalPages = computed(() => Math.max(1, Math.ceil(props.total / props.pageSize)))
    return () => {
      if (totalPages.value <= 1) return null
      return h('div', { class: 'pagination' }, [
        h('button', { class: 'page-btn', disabled: props.page <= 1, onClick: () => emit('change', props.page - 1) }, '‹'),
        h('span', { class: 'page-info' }, `${props.page} / ${totalPages.value}`),
        h('button', { class: 'page-btn', disabled: props.page >= totalPages.value, onClick: () => emit('change', props.page + 1) }, '›'),
      ])
    }
  }
})

const PAGE_SIZE = 50
const REQUIRED_COLUMNS = [
  'Flight.No', 'Airport.From', 'Airport.To',
  'Scheduled.Departure', 'Scheduled.Arrival',
  'Departure', 'Arrival', 'Distance.In.Meters',
  'Longitude.From', 'Latitude.From', 'Longitude.To', 'Latitude.To'
]

const form = reactive({
  file: null, airport: '', nRunways: null, safetyInterval: null,
  dateRange: { start: '', end: '' },
})

const isRunning      = ref(false)
const finalState     = ref('')
const steps          = ref([])
const summary        = ref(null)
const schedule       = ref([])
const removedFlights = ref([])

function pushStep(message, isError = false) {
  steps.value.push({ message, isError })
}

const scheduleSearch   = ref('')
const scheduleOpFilter = ref('')
const sort             = reactive({ col: 'scheduled_time', asc: true })
const schedulePage     = ref(1)

const filteredSchedule = computed(() => {
  let rows = schedule.value
  const q = scheduleSearch.value.trim().toLowerCase()
  if (q) rows = rows.filter(r => String(r.flight_id).toLowerCase().includes(q) || String(r.runway).toLowerCase().includes(q))
  if (scheduleOpFilter.value) rows = rows.filter(r => r.operation === scheduleOpFilter.value)
  return [...rows].sort((a, b) => {
    const va = a[sort.col], vb = b[sort.col]
    if (va < vb) return sort.asc ? -1 : 1
    if (va > vb) return sort.asc ? 1 : -1
    return 0
  })
})

const pagedSchedule = computed(() => {
  const s = (schedulePage.value - 1) * PAGE_SIZE
  return filteredSchedule.value.slice(s, s + PAGE_SIZE)
})

function sortBy(col) {
  if (sort.col === col) sort.asc = !sort.asc
  else { sort.col = col; sort.asc = true }
  schedulePage.value = 1
}

const removedSearch = ref('')
const removedPage   = ref(1)

const filteredRemoved = computed(() => {
  const q = removedSearch.value.trim().toLowerCase()
  if (!q) return removedFlights.value
  return removedFlights.value.filter(r =>
    r.flight_no.toLowerCase().includes(q) ||
    r.airport_from.toLowerCase().includes(q) ||
    r.airport_to.toLowerCase().includes(q)
  )
})

const pagedRemoved = computed(() => {
  const s = (removedPage.value - 1) * PAGE_SIZE
  return filteredRemoved.value.slice(s, s + PAGE_SIZE)
})

function fmtTime(iso) {
  if (!iso) return '-'
  try { return new Date(iso).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) }
  catch { return iso }
}

function delayClass(min) {
  if (min <= 0)  return 'delay--ok'
  if (min <= 10) return 'delay--minor'
  if (min <= 30) return 'delay--moderate'
  return 'delay--severe'
}

function checkHeaders(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = e => {
      const headers = e.target.result.split('\n')[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''))
      const missing = REQUIRED_COLUMNS.filter(c => !headers.includes(c))
      missing.length === 0 ? resolve() : reject(`缺少列：${missing.join(', ')}`)
    }
    reader.onerror = () => reject('文件读取失败')
    reader.readAsText(file.slice(0, 4096))
  })
}

async function onSubmit() {
  isRunning.value = true
  finalState.value = ''
  steps.value = []
  summary.value = null
  schedule.value = []
  removedFlights.value = []
  schedulePage.value = 1
  removedPage.value = 1

  pushStep('正在检查文件列头格式…')
  try {
    await checkHeaders(form.file)
    steps.value[steps.value.length - 1].message = '文件列头格式检查通过'
  } catch (msg) {
    steps.value[steps.value.length - 1].message = msg
    steps.value[steps.value.length - 1].isError = true
    isRunning.value = false
    finalState.value = 'error'
    return
  }

  try {
    const result = await uploadDataset(
      form.file,
      {
        airport:        form.airport || 'SBGR',
        nRunways:       form.nRunways || 3,
        safetyInterval: form.safetyInterval || 3,
        startDate:      form.dateRange.start,
        endDate:        form.dateRange.end,
      },
      (_step, message) => pushStep(message)
    )
    pushStep('调度完成 ✓')
    summary.value        = result.summary
    schedule.value       = result.schedule || []
    removedFlights.value = result.removed_flights || []
    finalState.value     = 'done'
  } catch (err) {
    pushStep(err.message, true)
    finalState.value = 'error'
  } finally {
    isRunning.value = false
  }
}
</script>

<style scoped>
.page { min-height: 100vh; display: flex; flex-direction: column; }

.header { background: #fff; border-bottom: 1px solid #eaecf0; position: sticky; top: 0; z-index: 10; }
.header__inner { max-width: 1200px; margin: 0 auto; padding: 0 32px; height: 60px; display: flex; align-items: center; justify-content: space-between; }
.header__brand { display: flex; align-items: center; gap: 12px; }
.header__logo { width: 32px; height: 32px; border-radius: 8px; background: linear-gradient(135deg, #4f6ef7, #7c3aed); display: flex; align-items: center; justify-content: center; overflow: hidden; }
.header__logo-img { width: 22px; height: 22px; object-fit: contain; }
.header__title { font-size: 16px; font-weight: 600; color: #1a1a2e; letter-spacing: -.2px; }
.header__badge { font-size: 11px; font-weight: 600; color: #4f6ef7; background: #eef1ff; padding: 2px 8px; border-radius: 20px; letter-spacing: .5px; }

.main { flex: 1; max-width: 1200px; margin: 0 auto; padding: 48px 32px 64px; width: 100%; }
.hero { margin-bottom: 40px; }
.hero__heading { font-size: 28px; font-weight: 600; color: #1a1a2e; letter-spacing: -.5px; margin-bottom: 8px; }
.hero__sub { font-size: 15px; color: #8a94a6; }

.card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }
@media (max-width: 760px) { .card-grid { grid-template-columns: 1fr; } }
.card { background: #fff; border: 1px solid #eaecf0; border-radius: 16px; padding: 28px; }
.card__head { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }
.card__num { font-size: 11px; font-weight: 700; color: #4f6ef7; background: #eef1ff; padding: 2px 7px; border-radius: 6px; letter-spacing: .5px; }
.card__title { font-size: 15px; font-weight: 600; color: #2d3a4a; }

.params { display: flex; flex-direction: column; gap: 18px; }
.param-row { display: flex; flex-direction: column; gap: 6px; }
.param-label { font-size: 12.5px; font-weight: 500; color: #6b7280; letter-spacing: .2px; }
.param-input { border: 1.5px solid #e4e7ec; border-radius: 8px; padding: 9px 13px; font-size: 14px; font-family: inherit; color: #2d3a4a; background: #fff; outline: none; transition: border-color .2s; }
.param-input:focus { border-color: #4f6ef7; }
.param-input::placeholder { color: #c4cad4; }

.submit-row { display: flex; align-items: center; gap: 16px; }
.btn-submit { display: inline-flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #4f6ef7, #7c3aed); color: #fff; border: none; border-radius: 10px; padding: 12px 28px; font-size: 15px; font-weight: 500; font-family: inherit; cursor: pointer; transition: opacity .2s, transform .1s; }
.btn-submit:hover:not(:disabled) { opacity: .9; transform: translateY(-1px); }
.btn-submit:disabled { opacity: .4; cursor: not-allowed; }
.submit-hint { font-size: 13px; color: #b0b8c9; }

.progress-box { margin-top: 24px; display: flex; flex-direction: column; gap: 6px; }
.progress-step { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 8px; background: #f8f9fb; border: 1px solid #eaecf0; font-size: 14px; color: #6b7280; }
.progress-step--active { background: #f0f3ff; border-color: #c7d2fe; color: #2d3a4a; }
.progress-step--done   { background: #f0fdf4; border-color: #bbf7d0; color: #15803d; }
.progress-step--error  { background: #fff1f2; border-color: #fecdd3; color: #be123c; }
.step-icon { width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; flex-shrink: 0; }
.step-msg { flex: 1; }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #c7d2fe; border-top-color: #4f6ef7; border-radius: 50%; animation: spin .7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.summary-box { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; font-size: 13px; color: #6b7280; }
.summary-box span { background: #fff; border: 1px solid #e4e7ec; border-radius: 6px; padding: 4px 12px; }

.result-section { margin-top: 40px; }
.result-head { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
.result-title { font-size: 17px; font-weight: 600; color: #1a1a2e; }
.result-count { font-size: 13px; color: #8a94a6; background: #f3f4f6; border-radius: 20px; padding: 2px 10px; }
.removed-count { background: #fff1f2; color: #e11d48; }
.result-filters { display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }
.filter-input { border: 1.5px solid #e4e7ec; border-radius: 8px; padding: 7px 12px; font-size: 13px; font-family: inherit; outline: none; transition: border-color .2s; min-width: 180px; }
.filter-input:focus { border-color: #4f6ef7; }
.filter-select { border: 1.5px solid #e4e7ec; border-radius: 8px; padding: 7px 12px; font-size: 13px; font-family: inherit; outline: none; background: #fff; cursor: pointer; }

.table-wrap { overflow-x: auto; border-radius: 12px; border: 1px solid #eaecf0; }
.data-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.data-table thead { background: #f8f9fb; }
.data-table th { padding: 11px 14px; text-align: left; font-weight: 600; color: #6b7280; font-size: 12px; letter-spacing: .3px; border-bottom: 1px solid #eaecf0; white-space: nowrap; }
.data-table td { padding: 10px 14px; border-bottom: 1px solid #f3f4f6; color: #2d3a4a; }
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover { background: #fafbff; }
.sortable { cursor: pointer; user-select: none; }
.sortable:hover { color: #4f6ef7; }
.sort-icon { color: #b0b8c9; font-size: 11px; }
.mono { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12.5px; }
.empty-row { text-align: center; color: #b0b8c9; padding: 24px; }

.badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11.5px; font-weight: 600; }
.badge--dep { background: #eff6ff; color: #2563eb; }
.badge--arr { background: #f0fdf4; color: #16a34a; }
.runway-badge { display: inline-block; padding: 2px 8px; border-radius: 6px; background: #f3f4f6; color: #4b5563; font-size: 12px; font-weight: 600; }

.delay { font-weight: 600; font-size: 13px; }
.delay--ok       { color: #16a34a; }
.delay--minor    { color: #ca8a04; }
.delay--moderate { color: #ea580c; }
.delay--severe   { color: #dc2626; }

.pagination { display: flex; align-items: center; justify-content: center; gap: 12px; margin-top: 16px; }
.page-btn { border: 1.5px solid #e4e7ec; background: #fff; border-radius: 8px; padding: 6px 14px; font-size: 15px; cursor: pointer; transition: border-color .2s; }
.page-btn:hover:not(:disabled) { border-color: #4f6ef7; color: #4f6ef7; }
.page-btn:disabled { opacity: .35; cursor: not-allowed; }
.page-info { font-size: 13px; color: #6b7280; }

.footer { text-align: center; padding: 20px; font-size: 12px; color: #c4cad4; }
</style>
