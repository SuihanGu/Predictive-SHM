<template>
  <div class="monitor-container">
    <!-- 工具栏 -->
    <div class="toolbar">
      <el-button type="primary" @click="refreshData">刷新数据</el-button>
      <span class="refresh-info">
        自动刷新：每 {{ refreshIntervalSec }} 秒 | 最后更新: {{ lastUpdate }}
      </span>
    </div>

    <!-- 告警通过对应图表外框红色显示，不再使用顶部横幅 -->
    <div v-for="(row, rowIdx) in sensorRows" :key="rowIdx" :class="['chart-row', row.length === 1 && row[0]?.full_width ? 'single' : '']">
      <SensorChart
        v-for="sensor in row"
        :key="sensor.key"
        :sensor="sensor"
        :chart-option="chartOptions[sensor.key]"
        :model="modelBySensor[sensor.key]"
        :available-models="availableModels"
        :get-model-label="getModelLabel"
        :threshold="thresholds[sensor.key] || { static: 0, residual: 0 }"
        :show-model-select="!!sensor.full_width"
        :has-alert="sensorsInAlert.has(sensor.key)"
        @update:model="(v) => { modelBySensor[sensor.key] = v; onModelChange(sensor.key) }"
        @update:threshold="onThresholdChange"
      />
    </div>

    <el-empty v-if="!sensorsConfig.length && !loading" description="暂无传感器配置，请检查 backend/config/monitor_config.json" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import axios from 'axios'
import SensorChart from '@/components/SensorChart.vue'

interface SensorConfig {
  key: string
  label: string
  unit: string
  data_key?: string
  channels?: string[]
  default_static_threshold?: number
  default_residual_threshold?: number
  default_threshold?: number
  step?: number
  precision?: number
  full_width?: boolean
}

interface ThresholdPair {
  static: number
  residual: number
}

const loading = ref(true)
const sensorsConfig = ref<SensorConfig[]>([])

// 后端无配置时的默认：测缝计
const DEFAULT_SENSORS: SensorConfig[] = [
  { key: 'crack', label: '测缝计', unit: 'mm', data_key: 'crack', default_static_threshold: 0.8, default_residual_threshold: 0.1, step: 0.1, precision: 2, full_width: true },
]
const modelsConfig = ref<Array<{ name: string; label?: string; description?: string }>>([])
const availableModels = ref<string[]>(['transformer_cnn'])
const modelBySensor = ref<Record<string, string>>({})
const thresholds = ref<Record<string, ThresholdPair>>({})
const chartOptions = ref<Record<string, object>>({})
const predictionBySensor = ref<Record<string, number[] | number[][]>>({})
const alerts = ref<any[]>([])
const lastUpdate = ref('')
const historyData = ref<any[]>([])
const refreshIntervalSec = ref(60)
let refreshTimer: ReturnType<typeof setInterval> | null = null
let thresholdSyncTimer: ReturnType<typeof setTimeout> | null = null

/**
 * 将传感器按 full_width 分组为行：full_width 单独一行，其余每两个并排。
 * 实际工程中布局由 monitor_config.json 的 full_width 决定，可按需调整。
 */
const sensorRows = computed(() => {
  const sensors = sensorsConfig.value
  const rows: SensorConfig[][] = []
  let halfBuffer: SensorConfig[] = []
  for (const s of sensors) {
    if (s.full_width) {
      if (halfBuffer.length) {
        rows.push([...halfBuffer])
        halfBuffer = []
      }
      rows.push([s])
    } else {
      halfBuffer.push(s)
      if (halfBuffer.length === 2) {
        rows.push([...halfBuffer])
        halfBuffer = []
      }
    }
  }
  if (halfBuffer.length) rows.push(halfBuffer)
  return rows
})

/** 当前有告警的传感器 key 集合，用于图表外框高亮。
 * 完全由最新一次 /api/alerts/check 结果驱动：告警结束时该 sensor 不在列表中，
 * sensorsInAlert 不包含其 key，对应图表的 hasAlert 变为 false，外框恢复默认样式。 */
const sensorsInAlert = computed(() => new Set((alerts.value || []).map((a: { type?: string }) => a.type).filter(Boolean)))
function mockPrediction(_sensorKey: string, lastVal: number): number[] {
  const base = lastVal + 0.01 * (Math.random() - 0.5)
  return Array.from({ length: 6 }, (_, i) => base + 0.005 * i + (Math.random() - 0.5) * 0.02)
}

/** 为多通道分别生成 mock 预测，每个通道 6 步 */
function mockPredictionMulti(lastVals: number[]): number[][] {
  return lastVals.map((v) => mockPrediction('', v))
}

const CHART_COLORS = ['#5470C6', '#91CC75', '#FAC858', '#EE6666', '#73C0DE', '#3BA272', '#FC8452', '#9A60B4']

function createChartOption(
  _title: string,
  hist: Array<[number, number]>,
  pred: Array<[number, number]>,
  unit = '',
  threshold?: number,
  precision = 3
) {
  return createChartOptionMulti(
    [{ name: '实测', data: hist }],
    [{ name: '预测', data: pred }],
    unit,
    threshold,
    precision
  )
}

/** 创建多通道图表的 ECharts 配置（多条实测 + 预测曲线，首条可显示静态阈值）
 * 根据数据范围设定 Y 轴 min/max，使变化幅度更直观；支持 precision 控制小数位 */
function createChartOptionMulti(
  histSeries: Array<{ name: string; data: Array<[number, number]> }>,
  predSeries: Array<{ name: string; data: Array<[number, number]> }>,
  unit = '',
  threshold?: number,
  precision = 3
) {
  const series: any[] = []
  const formatValue = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(precision))
  histSeries.forEach((s, i) => {
    series.push({
      name: s.name,
      type: 'line' as const,
      data: s.data,
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, color: CHART_COLORS[i % CHART_COLORS.length] },
      itemStyle: { color: CHART_COLORS[i % CHART_COLORS.length] },
      markLine:
        threshold != null && threshold > 0 && i === 0
          ? { silent: true, data: [{ yAxis: threshold, name: '静态阈值', lineStyle: { type: 'dashed', color: '#F56C6C' } }] }
          : undefined,
    })
  })
  predSeries.forEach((s, i) => {
    series.push({
      name: s.name,
      type: 'line' as const,
      data: s.data,
      smooth: true,
      showSymbol: false,
      lineStyle: { type: 'dashed', width: 2, color: CHART_COLORS[(histSeries.length + i) % CHART_COLORS.length] },
      itemStyle: { color: CHART_COLORS[(histSeries.length + i) % CHART_COLORS.length] },
    })
  })
  const allX = series.reduce<number[]>((acc, s) => acc.concat(s.data.map((d: [number, number]) => d[0])), []).filter(Boolean)
  const minT = allX.length ? Math.min(...allX) : undefined
  const maxT = allX.length ? Math.max(...allX) : undefined

  // 根据数据范围设定 Y 轴，使变化幅度直观可见；对小范围数据强制最小跨度（仅用系列数据，不含阈值，避免阈值拉大范围压平曲线）
  const allY = series.reduce<number[]>((acc, s) => acc.concat(s.data.map((d: [number, number]) => d[1])), []).filter((v) => v != null && !Number.isNaN(v))
  let yMin: number | undefined
  let yMax: number | undefined
  if (allY.length) {
    const dataMin = Math.min(...allY)
    const dataMax = Math.max(...allY)
    const dataCenter = (dataMin + dataMax) / 2
    const span = Math.max(dataMax - dataMin, 0)
    // 最小 Y 轴跨度：至少为数据中心的 25%，或至少 0.05（测缝计等小数值时变化更明显）
    const minSpan = Math.max(span, Math.abs(dataCenter) * 0.25, 0.05)
    const halfSpan = minSpan / 2
    yMin = dataCenter - halfSpan
    yMax = dataCenter + halfSpan
    // 若设置了静态阈值且落在当前范围外，仅当阈值与范围较近时才扩大（避免阈值 0.8 把 0.1x 的数据压成一条线）
    if (threshold != null && threshold > 0) {
      const pad = halfSpan * 0.3
      const maxReach = Math.max(minSpan * 2, 0.15)
      if (threshold < yMin && yMin - threshold < maxReach) yMin = Math.min(threshold - pad, yMin)
      if (threshold > yMax && threshold - yMax < maxReach) yMax = Math.max(threshold + pad, yMax)
    }
  }

  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'line' },
      valueFormatter: (value: number) => formatValue(value),
    },
    legend: { data: series.map((s) => s.name), bottom: 8 },
    grid: { top: 40, right: 30, bottom: 80, left: 60, containLabel: true },
    xAxis: {
      type: 'time',
      boundaryGap: false,
      min: minT,
      max: maxT,
      axisLabel: {
        formatter: (v: number) => {
          const d = new Date(v)
          const pad = (n: number) => String(n).padStart(2, '0')
          return `${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
        },
      },
    },
    yAxis: {
      type: 'value',
      name: unit,
      nameTextStyle: { fontSize: 11 },
      min: yMin,
      max: yMax,
      splitNumber: 6,
      axisLabel: { formatter: (v: number) => formatValue(v) },
    },
    dataZoom: [
      { type: 'inside', start: 70, end: 100 },
      { type: 'slider', start: 70, end: 100, height: 20, bottom: 10 },
    ],
    series,
  }
}

/** 将历史记录转为单条时序数据 [[时间戳, 值], ...]，支持单通道或取多通道首通道 */
function toSeries(records: any[], key: string): Array<[number, number]> {
  const sensor = sensorsConfig.value.find((s) => s.key === key)
  const channels = sensor?.channels
  if (channels?.length) {
    return toSeriesMulti(records, channels)[0]?.data ?? []
  }
  const dk = sensor?.data_key || key
  return records.map((r) => {
    const ts = r.timestamp
    const ms = typeof ts === 'number' && ts < 1e12 ? ts * 1000 : ts
    const v = r[dk] ?? r[key] ?? r.data1 ?? r.data2 ?? 0
    return [ms ?? Date.now(), Number(v)]
  })
}

/** 将历史记录按多通道转为多条时序数据，每条 { name, data: [[时间戳, 值], ...] } */
function toSeriesMulti(
  records: any[],
  channels: string[]
): Array<{ name: string; data: Array<[number, number]> }> {
  return channels.map((ch) => ({
    name: ch,
    data: records.map((r) => {
      const ts = r.timestamp
      const ms = typeof ts === 'number' && ts < 1e12 ? ts * 1000 : ts
      const v = r[ch] ?? r.data1 ?? r.data2 ?? 0
      return [ms ?? Date.now(), Number(v)]
    }),
  }))
}

/** 将预测数组转为时序点 [[时间戳, 值], ...]，支持单通道或取多通道首通道 */
function predToSeries(
  pred: number[] | number[][],
  stepMs = 60000,
  lastHistTimestampMs?: number
): Array<[number, number]> {
  const arr = Array.isArray(pred?.[0]) && typeof pred[0][0] === 'number'
    ? (pred as number[][])[0]
    : (pred as number[]) ?? []
  const base = lastHistTimestampMs != null ? lastHistTimestampMs + 1 : Date.now()
  return arr.map((v, i) => [base + i * stepMs, Number(v)])
}

/** 将多通道预测转为多条时序数据，每条 { name: `预测-通道名`, data } */
function predToSeriesMulti(
  pred: number[] | number[][],
  channelNames: string[],
  stepMs = 60000,
  lastHistTimestampMs?: number
): Array<{ name: string; data: Array<[number, number]> }> {
  const isMulti = Array.isArray(pred) && pred.length > 0 && Array.isArray((pred as number[][])[0])
  const arrs = isMulti ? (pred as number[][]) : [(pred as number[]) ?? []]
  const base = lastHistTimestampMs != null ? lastHistTimestampMs + 1 : Date.now()
  return arrs.map((arr, i) => ({
    name: `预测-${channelNames[i] ?? i + 1}`,
    data: (arr ?? []).map((v, j) => [base + j * stepMs, Number(v)]),
  }))
}

/** 拉取监控配置（传感器、模型）及后端阈值，初始化各传感器状态 */
async function fetchConfig() {
  loading.value = true
  try {
    const res = await axios.get<{ sensors?: SensorConfig[]; models?: any[] }>('/api/config/monitor')
    let sensors = res.data?.sensors ?? []
    const models = res.data?.models ?? []
    if (!sensors.length) {
      sensors = DEFAULT_SENSORS
      modelsConfig.value = [{ name: 'transformer_cnn', label: 'Transformer-CNN' }]
    } else {
      modelsConfig.value = models
    }
    sensorsConfig.value = sensors

    const modelsRes = await axios.get<{ available_models?: string[] }>('/api/models/list')
    const list = modelsRes.data?.available_models
    if (Array.isArray(list) && list.length) availableModels.value = list

    modelBySensor.value = {}
    thresholds.value = {}
    predictionBySensor.value = {}
    for (const s of sensors) {
      const k = s.key
      modelBySensor.value[k] = availableModels.value[0] || 'transformer_cnn'
      thresholds.value[k] = {
        static: s.default_static_threshold ?? s.default_threshold ?? 0.8,
        residual: s.default_residual_threshold ?? s.default_threshold ?? 0.1,
      }
      predictionBySensor.value[k] = []
    }
    try {
      const thRes = await axios.get<Record<string, ThresholdPair>>('/api/alerts/thresholds')
      const backendTh = thRes.data
      if (backendTh && typeof backendTh === 'object') {
        for (const k of Object.keys(thresholds.value)) {
          const v = backendTh[k]
          if (v && typeof v === 'object') {
            if (v.static != null) thresholds.value[k].static = v.static
            if (v.residual != null) thresholds.value[k].residual = v.residual
          }
        }
      }
    } catch (_) {}
  } catch (e) {
    console.error('配置加载失败，使用默认测缝计', e)
    sensorsConfig.value = DEFAULT_SENSORS
    modelsConfig.value = [{ name: 'transformer_cnn', label: 'Transformer-CNN' }]
    availableModels.value = ['transformer_cnn']
    modelBySensor.value = {}
    thresholds.value = {}
    predictionBySensor.value = {}
    for (const s of DEFAULT_SENSORS) {
      modelBySensor.value[s.key] = 'transformer_cnn'
      thresholds.value[s.key] = {
        static: s.default_static_threshold ?? s.default_threshold ?? 0.8,
        residual: s.default_residual_threshold ?? s.default_threshold ?? 0.1,
      }
      predictionBySensor.value[s.key] = []
    }
    try {
      const thRes = await axios.get<Record<string, ThresholdPair>>('/api/alerts/thresholds')
      const backendTh = thRes.data
      if (backendTh && typeof backendTh === 'object') {
        for (const k of Object.keys(thresholds.value)) {
          const v = backendTh[k]
          if (v && typeof v === 'object') {
            if (v.static != null) thresholds.value[k].static = v.static
            if (v.residual != null) thresholds.value[k].residual = v.residual
          }
        }
      }
    } catch (_) {}
  } finally {
    loading.value = false
  }
}

/** 刷新数据：拉取历史、调用预测 API、检查预警、更新图表 */
const refreshData = async () => {
  if (!sensorsConfig.value.length) return
  try {
    let history: any[] = []
    try {
      const res = await axios.get('/api/data/sample')
      history = Array.isArray(res.data) ? res.data : (res.data?.data ?? res.data?.records ?? [])
    } catch {
      history = []
    }
    if (!history.length) {
      const now = Math.floor(Date.now() / 1000)
      const sensors = sensorsConfig.value
      history = Array.from({ length: 144 }, (_, i) => {
        const t = now - (144 - i) * 600
        const row: Record<string, number> = { timestamp: t }
        const base = 0.1 + 0.02 * (i % 10)
        for (const s of sensors) {
          if (s.channels?.length) {
            s.channels.forEach((ch, j) => { row[ch] = base + j * 0.01 })
          } else {
            row[s.data_key || s.key] = base
          }
        }
        return row
      })
    }
    historyData.value = history

    const last = history[history.length - 1] || {}
    const pred: Record<string, number[] | number[][]> = {}
    const crackSensor = sensorsConfig.value.find((s) => s.key === 'crack')
    const crackChannels = crackSensor?.channels ?? ['crack_1']

    // 仅当该传感器有预测模型下拉框（full_width）且已选模型时，才调用预测 API 并展示预测曲线
    if (crackSensor?.full_width && modelBySensor.value.crack) {
      try {
        await axios.post(`/api/models/switch?model_name=${encodeURIComponent(modelBySensor.value.crack)}`)
        const predictRes = await axios.post('/api/predict', {
          history_data: history,
          model_name: modelBySensor.value.crack,
        })
        const p = predictRes.data?.prediction
        // 新版后端：prediction = { time_index, readings, sensor_ids, ... }
        if (p && Array.isArray(p.readings)) {
          const steps = p.readings as number[][]
          const crackPred: number[][] = crackChannels.map((_, j) => steps.map((s) => Number(s?.[j] ?? s?.[0] ?? 0)))
          pred.crack = crackPred.length ? crackPred : crackChannels.map(() => mockPrediction('crack', last.crack_1 ?? last.crack ?? 0.12))
        } else if (Array.isArray(p?.[0]) && Array.isArray((p as number[][])[0]?.[0])) {
          // 旧版后端：prediction = [ [ [step][dim] ] ]（batch 维）
          const steps = p[0] as number[][]
          const crackPred: number[][] = crackChannels.map((_, j) => steps.map((s) => Number(s[j] ?? s[0])))
          pred.crack = crackPred.length ? crackPred : crackChannels.map(() => mockPrediction('crack', last.crack_1 ?? last.crack ?? 0.12))
        } else {
          // 旧版后端：prediction = [step] 或 [ [step] ]
          const flat = Array.isArray(p?.[0]) ? (p[0] as number[]) : (Array.isArray(p) ? p : [])
          pred.crack = flat.length ? [flat] : [mockPrediction('crack', last.crack_1 ?? last.crack ?? 0.12)]
        }
      } catch {
        pred.crack = crackChannels.map(() => mockPrediction('crack', last.crack_1 ?? last.crack ?? 0.12))
      }
    }

    for (const s of sensorsConfig.value) {
      if (pred[s.key]) continue
      // 无预测模型下拉框的传感器不展示预测曲线
      if (!s.full_width) {
        pred[s.key] = s.channels?.length ? s.channels.map(() => []) : []
        continue
      }
      const channels = s.channels
      if (channels?.length) {
        const lastVals = channels.map((ch) => Number(last[ch] ?? last[s.data_key ?? s.key] ?? 0.1))
        pred[s.key] = mockPredictionMulti(lastVals)
      } else {
        const dk = s.data_key || s.key
        const v = last[dk] ?? last[s.key] ?? last.data1 ?? last.data2 ?? 0.1
        pred[s.key] = mockPrediction(s.key, Number(v))
      }
    }
    predictionBySensor.value = pred

    try {
      const sensorKeys = sensorsConfig.value.map((s) => s.key)
      const rows = history.slice(-60).map((r: any) =>
        sensorKeys.map((k) => {
          const sensor = sensorsConfig.value.find((x) => x.key === k)
          const ch = sensor?.channels?.[0] ?? sensor?.data_key ?? k
          return r[ch] ?? r[k] ?? r.data1 ?? 0
        })
      )
      while (rows.length < 60) rows.unshift(sensorKeys.map(() => 0))
      const predVals = sensorKeys.map((k) => {
        const p = predictionBySensor.value[k]
        if (!p) return 0
        const first = (p as number[] | number[][])[0]
        return Array.isArray(first) ? (first[0] ?? 0) : (first ?? 0)
      })
      const alertRes = await axios.post('/api/alerts/check', {
        history: [rows],
        prediction: [predVals],
        sensor_keys: sensorKeys,
      })
      alerts.value = alertRes.data?.alerts ?? []
    } catch {
      alerts.value = []
    }

    lastUpdate.value = new Date().toLocaleString()
    await nextTick()
    updateAllCharts()
  } catch (e) {
    console.error('数据刷新失败', e)
    updateAllCharts()
  }
}

/** 根据历史与预测数据，为每个传感器生成 ECharts 配置并写入 chartOptions */
function updateAllCharts() {
  const hist = historyData.value
  const pred = predictionBySensor.value
  // 后端统一按 10min 重采样并预测（ULDM），预测点间隔与历史一致
  const stepMs = 10 * 60 * 1000
  const opts: Record<string, object> = {}

  for (const s of sensorsConfig.value) {
    const channels = s.channels
    const staticTh = thresholds.value[s.key]?.static ?? 0
    const thresholdOpt = staticTh > 0 ? staticTh : undefined

    if (channels?.length) {
      const histSeries = toSeriesMulti(hist, channels)
      const lastTs = histSeries[0]?.data?.length ? histSeries[0].data[histSeries[0].data.length - 1][0] : undefined
      const predRaw = pred[s.key] ?? []
      const predSeries = predToSeriesMulti(predRaw, channels, stepMs, lastTs)
      const precision = s.precision ?? 3
      opts[s.key] = createChartOptionMulti(histSeries, predSeries, s.unit, thresholdOpt, precision)
    } else {
      const histSeries = toSeries(hist, s.key)
      const lastTs = histSeries.length ? histSeries[histSeries.length - 1][0] : undefined
      const predSeries = predToSeries(pred[s.key] || [], stepMs, lastTs)
      const precision = s.precision ?? 3
      opts[s.key] = createChartOption('', histSeries, predSeries, s.unit, thresholdOpt, precision)
    }
  }
  chartOptions.value = opts
}

/** 根据模型名称返回显示标签 */
function getModelLabel(name: string) {
  return modelsConfig.value.find((x) => x.name === name)?.label || name
}

/** 模型切换时重新拉取数据并刷新图表 */
function onModelChange(_sensor: string) {
  refreshData()
}

/** 阈值变更时更新本地状态、刷新图表，并防抖同步到后端 */
function onThresholdChange(payload: { key: string; static?: number; residual?: number }) {
  const k = payload.key
  if (!thresholds.value[k]) thresholds.value[k] = { static: 0, residual: 0 }
  if (payload.static != null) thresholds.value[k].static = payload.static
  if (payload.residual != null) thresholds.value[k].residual = payload.residual
  updateAllCharts()
  syncThresholdsToBackendDebounced()
}

/** 500ms 防抖后调用 syncThresholdsToBackend，避免频繁请求 */
function syncThresholdsToBackendDebounced() {
  if (thresholdSyncTimer) clearTimeout(thresholdSyncTimer)
  thresholdSyncTimer = setTimeout(() => {
    syncThresholdsToBackend()
    thresholdSyncTimer = null
  }, 500)
}

/** 将当前阈值 POST 到 /api/alerts/thresholds 持久化 */
async function syncThresholdsToBackend() {
  try {
    await axios.post('/api/alerts/thresholds', thresholds.value)
  } catch (e) {
    console.error('阈值同步失败', e)
  }
}

onMounted(async () => {
  await fetchConfig()
  await nextTick()
  refreshData()
  refreshTimer = setInterval(refreshData, refreshIntervalSec.value * 1000)
})

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer)
  if (thresholdSyncTimer) clearTimeout(thresholdSyncTimer)
})
</script>

<style scoped>
.monitor-container {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: 100vh;
}
.toolbar {
  margin-bottom: 20px;
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.refresh-info {
  margin-left: auto;
  color: #606266;
  font-size: 13px;
}
.chart-row {
  display: flex;
  gap: 20px;
}
.chart-row.single {
  display: block;
}
@media (max-width: 1200px) {
  .chart-row {
    flex-direction: column;
  }
}
</style>
