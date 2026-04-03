<template>
  <div class="drp">
    <div class="drp__field">
      <label class="drp__label">开始日期</label>
      <input
        type="date"
        class="drp__input"
        :value="modelValue.start"
        @change="e => emit('update:modelValue', { ...modelValue, start: e.target.value })"
      />
    </div>
    <div class="drp__sep">—</div>
    <div class="drp__field">
      <label class="drp__label">结束日期</label>
      <input
        type="date"
        class="drp__input"
        :value="modelValue.end"
        @change="e => emit('update:modelValue', { ...modelValue, end: e.target.value })"
      />
    </div>
  </div>
</template>

<script setup>
defineProps({ modelValue: { type: Object, default: () => ({ start: '', end: '' }) } })
const emit = defineEmits(['update:modelValue'])
</script>

<style scoped>
.drp { display: flex; align-items: flex-end; gap: 10px; }
.drp__field { display: flex; flex-direction: column; gap: 5px; flex: 1; }
.drp__label { font-size: 12px; color: #8a94a6; font-weight: 500; letter-spacing: .3px; }
.drp__input {
  border: 1.5px solid #e4e7ec;
  border-radius: 8px;
  padding: 9px 12px;
  font-size: 14px;
  font-family: inherit;
  color: #2d3a4a;
  background: #fff;
  outline: none;
  transition: border-color .2s;
  width: 100%;
}
.drp__input:focus { border-color: #4f6ef7; }
.drp__sep { color: #b0b8c9; padding-bottom: 10px; font-size: 16px; }
</style>
