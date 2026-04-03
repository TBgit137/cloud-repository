<template>
  <div
    class="dropzone"
    :class="{ 'dropzone--active': isDragging, 'dropzone--filled': file }"
    @dragover.prevent="isDragging = true"
    @dragleave.prevent="isDragging = false"
    @drop.prevent="onDrop"
    @click="triggerInput"
  >
    <input ref="inputRef" type="file" accept=".csv" style="display:none" @change="onFileChange" />

    <template v-if="!file">
      <div class="dropzone__icon">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <p class="dropzone__title">拖拽 CSV 文件至此处</p>
      <p class="dropzone__sub">或 <span class="dropzone__link">点击选择文件</span></p>
      <p class="dropzone__hint">
        文件需包含列：<code>Flight.No</code> · <code>Airport.From</code> · <code>Airport.To</code> ·
        <code>Scheduled.Departure</code> · <code>Scheduled.Arrival</code> ·
        <code>Departure</code> · <code>Arrival</code> · <code>Distance.In.Meters</code> ·
        <code>Longitude.From/To</code> · <code>Latitude.From/To</code>
      </p>
    </template>

    <template v-else>
      <div class="dropzone__icon dropzone__icon--ok">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      </div>
      <p class="dropzone__title">{{ file.name }}</p>
      <p class="dropzone__sub">
        {{ fileSizeLabel }} &nbsp;·&nbsp;
        <span class="dropzone__link" @click.stop="clearFile">移除</span>
      </p>
    </template>
  </div>
</template>


<script setup>
import { ref, computed } from 'vue'

const emit = defineEmits(['update:file'])
const inputRef = ref(null)
const isDragging = ref(false)
const file = ref(null)

const fileSizeLabel = computed(() => {
  if (!file.value) return ''
  const kb = file.value.size / 1024
  return kb > 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(0)} KB`
})

function triggerInput() { inputRef.value?.click() }

function onFileChange(e) {
  const f = e.target.files[0]
  if (f) setFile(f)
}

function onDrop(e) {
  isDragging.value = false
  const f = e.dataTransfer.files[0]
  if (f && f.name.endsWith('.csv')) setFile(f)
}

function setFile(f) {
  file.value = f
  emit('update:file', f)
}

function clearFile() {
  file.value = null
  if (inputRef.value) inputRef.value.value = ''
  emit('update:file', null)
}
</script>

<style scoped>
.dropzone {
  border: 1.5px dashed #d0d5dd;
  border-radius: 14px;
  background: #fff;
  padding: 40px 24px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  transition: border-color .2s, background .2s;
  text-align: center;
  min-height: 220px;
  justify-content: center;
}
.dropzone--active { border-color: #4f6ef7; background: #f0f3ff; }
.dropzone--filled { border-style: solid; border-color: #4f6ef7; }
.dropzone__icon { color: #b0b8c9; margin-bottom: 4px; }
.dropzone__icon--ok { color: #4f6ef7; }
.dropzone__title { font-size: 15px; font-weight: 500; color: #2d3a4a; }
.dropzone__sub { font-size: 13px; color: #8a94a6; }
.dropzone__link { color: #4f6ef7; cursor: pointer; text-decoration: underline; }
.dropzone__hint {
  font-size: 11.5px;
  color: #b0b8c9;
  line-height: 1.7;
  max-width: 420px;
  margin-top: 4px;
}
.dropzone__hint code {
  font-family: 'SF Mono', monospace;
  background: #f3f4f6;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 11px;
}
</style>
