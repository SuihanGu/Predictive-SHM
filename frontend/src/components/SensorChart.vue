<template>
  <div :class="['chart-container', fullWidth ? 'full-width' : 'half-width', { 'chart-container--alert': hasAlert }]">
    <div class="chart-header">
      <span class="chart-title">{{ sensor.label }} ({{ sensor.unit }})</span>
      <div class="chart-controls">
        <template v-if="showModelSelect">
          <span class="control-label">Model</span>
          <el-select
            :model-value="model"
            size="small"
            :style="{ width: fullWidth ? '140px' : '120px' }"
            @update:model-value="emit('update:model', $event)"
          >
            <el-option
              v-for="m in availableModels"
              :key="m"
              :label="getModelLabel(m)"
              :value="m"
            />
          </el-select>
        </template>
        <span class="control-label">Static threshold</span>
        <el-input-number
          :model-value="threshold.static"
          :min="0"
          :step="step"
          :precision="precision"
          size="small"
          placeholder="0=disabled"
          class="control-input"
          @update:model-value="onThresholdChange('static', $event)"
        />
        <span class="control-label">Prediction residual</span>
        <el-input-number
          :model-value="threshold.residual"
          :min="0"
          :step="step"
          :precision="precision"
          size="small"
          placeholder="0=disabled"
          class="control-input"
          @update:model-value="onThresholdChange('residual', $event)"
        />
      </div>
    </div>
    <div ref="chartEl" class="chart"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  CanvasRenderer,
])

interface SensorConfig {
  key: string
  label: string
  unit: string
  data_key?: string
  channels?: string[]
  step?: number
  precision?: number
  full_width?: boolean
}

interface ThresholdPair {
  static: number
  residual: number
}

const props = withDefaults(
  defineProps<{
    sensor: SensorConfig
    chartOption?: object | null
    model?: string
    availableModels?: string[]
    getModelLabel?: (name: string) => string
    threshold: ThresholdPair
    showModelSelect?: boolean
    hasAlert?: boolean
  }>(),
  {
    chartOption: null,
    model: 'transformer_cnn',
    availableModels: () => ['transformer_cnn'],
    getModelLabel: (n: string) => n,
    showModelSelect: false,
    hasAlert: false,
  }
)

const emit = defineEmits<{
  (e: 'update:model', v: string): void
  (e: 'update:threshold', payload: { key: string; static?: number; residual?: number }): void
}>()

const chartEl = ref<HTMLElement | null>(null)
let chartInstance: echarts.ECharts | null = null

const fullWidth = computed(() => props.sensor.full_width ?? false)
const step = computed(() => props.sensor.step ?? 0.1)
const precision = computed(() => props.sensor.precision ?? 2)

function onThresholdChange(field: 'static' | 'residual', val: number) {
  emit('update:threshold', {
    key: props.sensor.key,
    [field]: val,
  })
}

watch(
  () => props.chartOption,
  (opt) => {
    if (chartInstance && opt) {
      chartInstance.setOption(opt as any, { notMerge: true })
      chartInstance.resize()
    }
  },
  { deep: true }
)

function resize() {
  chartInstance?.resize()
}

onMounted(() => {
  if (chartEl.value) {
    chartInstance = echarts.init(chartEl.value)
    if (props.chartOption) {
      chartInstance.setOption(props.chartOption as any, { notMerge: true })
    }
    window.addEventListener('resize', resize)
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', resize)
  chartInstance?.dispose()
  chartInstance = null
})

defineExpose({ resize })
</script>

<style scoped>
.chart-container {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  margin-bottom: 20px;
}
.chart-container.full-width {
  width: 100%;
}
.chart-container.half-width {
  flex: 1;
  min-width: 0;
}
/* 告警时显示红框；hasAlert 变为 false 时此类不应用，外框恢复 .chart-container 默认样式 */
.chart-container--alert {
  border: 2px solid #f56c6c;
  box-shadow: 0 0 0 1px #f56c6c;
}
.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 2px solid #409eff;
}
.chart-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}
.chart-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.chart-controls .control-label {
  font-size: 12px;
  color: #909399;
  margin-right: 2px;
}
.chart-controls .control-input {
  width: 100px;
}
.chart {
  width: 100%;
  height: 340px;
}
</style>
