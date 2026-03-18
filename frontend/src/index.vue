<template>
  <div class="shm-monitor-container">
    <div class="header">
      <h2>Structural Health Monitoring Data</h2>
      <div class="refresh-info">
        <span>Data updated: {{ lastUpdateTime }}</span>
        <el-button size="small" @click="refreshData">Refresh</el-button>
      </div>
    </div>

    <!-- Crack meter - full width -->
    <div class="chart-container full-width">
      <div class="chart-title">Crack meter</div>
      <div ref="crackChartRef" class="chart"></div>
    </div>

    <!-- 其他传感器 - 两列布局 -->
    <div class="chart-row">
      <!-- Tilt X -->
      <div class="chart-container half-width">
        <div class="chart-title">Tilt probe (X)</div>
        <div ref="tiltXChartRef" class="chart"></div>
      </div>

      <!-- Tilt Y -->
      <div class="chart-container half-width">
        <div class="chart-title">Tilt probe (Y)</div>
        <div ref="tiltYChartRef" class="chart"></div>
      </div>
    </div>

    <div class="chart-row">
      <!-- Settlement -->
      <div class="chart-container half-width">
        <div class="chart-title">Settlement</div>
        <div ref="levelChartRef" class="chart"></div>
      </div>

      <!-- Water level -->
      <div class="chart-container half-width">
        <div class="chart-title">Water level</div>
        <div ref="waterLevelChartRef" class="chart"></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import axios from "axios";

// 注册 ECharts 组件
echarts.use([
  LineChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
  CanvasRenderer
]);

// 接口定义
interface CrackDataPoint {
  id?: number;
  time?: string;
  timestamp?: string | number;
  channel?: string;
  code?: string;
  number?: string;
  data1?: number;
  data2?: number;
  data3?: number;
}

interface TiltDataPoint {
  id?: number;
  time?: string;
  timestamp?: string | number;
  code?: string;
  number?: string;
  data1?: number; // x方向
  data2?: number; // y方向
  data3?: number; // 温度
}

interface LevelDataPoint {
  id?: number;
  time?: string;
  timestamp?: string | number;
  number?: string;
  data1?: number; // 沉降
  data2?: number;
  data3?: number | null;
}

interface WaterLevelDataPoint {
  id?: number;
  time?: string;
  timestamp?: string | number;
  number?: string;
  data1?: number; // 水位(mm)
  data2?: number;
  data3?: number;
}

// 设备编号
const CRACK_NUMBERS = ["623622", "623628", "623641"];
const TILT_NUMBERS = ["00476464", "00476465", "00476466", "00476467"];
const LEVEL_NUMBERS = ["004521", "004548", "004591", "152947"];

// Chart 引用
const crackChartRef = ref<HTMLDivElement>();
const tiltXChartRef = ref<HTMLDivElement>();
const tiltYChartRef = ref<HTMLDivElement>();
const levelChartRef = ref<HTMLDivElement>();
const waterLevelChartRef = ref<HTMLDivElement>();

// Chart 实例
let crackChart: echarts.ECharts | null = null;
let tiltXChart: echarts.ECharts | null = null;
let tiltYChart: echarts.ECharts | null = null;
let levelChart: echarts.ECharts | null = null;
let waterLevelChart: echarts.ECharts | null = null;

// 数据刷新相关
const lastUpdateTime = ref<string>("");
let refreshTimer: number | null = null;
const REFRESH_INTERVAL = 10 * 60 * 1000; // 10分钟

// 获取当前时间戳（UNIX格式，秒）
function getCurrentTimestamp(): number {
  return Math.floor(Date.now() / 1000);
}

// 获取过去24小时的时间戳范围
function getTimeRange(): { timestamp1: number; timestamp2: number } {
  const now = getCurrentTimestamp();
  const dayAgo = now - 24 * 60 * 60; // 24小时前
  return {
    timestamp1: dayAgo,
    timestamp2: now
  };
}

// API 调用函数
async function fetchCrackData() {
  const { timestamp1, timestamp2 } = getTimeRange();
  const response = await axios.get<CrackDataPoint[]>(
    "http://139.159.136.213:4999/iem/shm/jmData",
    {
      params: { timestamp1, timestamp2 },
      timeout: 15000
    }
  );
  return Array.isArray(response.data) ? response.data : [];
}

async function fetchTiltData() {
  const { timestamp1, timestamp2 } = getTimeRange();
  const response = await axios.get<TiltDataPoint[]>(
    "http://139.159.136.213:4999/iem/shm/jmBus",
    {
      params: { timestamp1, timestamp2 },
      timeout: 15000
    }
  );
  return Array.isArray(response.data) ? response.data : [];
}

async function fetchLevelData() {
  const { timestamp1, timestamp2 } = getTimeRange();
  const response = await axios.get<LevelDataPoint[]>(
    "http://139.159.136.213:4999/iem/shm/jmLevel",
    {
      params: { timestamp1, timestamp2 },
      timeout: 15000
    }
  );
  return Array.isArray(response.data) ? response.data : [];
}

async function fetchWaterLevelData() {
  const { timestamp1, timestamp2 } = getTimeRange();
  const response = await axios.get<WaterLevelDataPoint[]>(
    "http://139.159.136.213:4999/iem/shm/jmWlg",
    {
      params: { timestamp1, timestamp2 },
      timeout: 15000
    }
  );
  return Array.isArray(response.data) ? response.data : [];
}

// 创建 ECharts 配置（参考堆叠折线图样式）
function createChartOption(
  title: string,
  seriesData: Array<{ name: string; data: Array<[number, number | null]> }>,
  unit: string = ""
) {
  const colors = ["#5470C6", "#91CC75", "#FAC858", "#EE6666", "#73C0DE", "#3BA272", "#FC8452", "#9A60B4"];
  
  const series = seriesData.map((item, idx) => ({
    name: item.name,
    type: "line",
    stack: "Total",
    showSymbol: false,
    data: item.data,
    areaStyle: {
      opacity: 0.3
    },
    lineStyle: {
      width: 2,
      color: colors[idx % colors.length]
    },
    itemStyle: {
      color: colors[idx % colors.length]
    },
    smooth: 0.4,
    connectNulls: true
  }));

  // 获取所有时间戳范围
  const allTimestamps: number[] = [];
  seriesData.forEach(item => {
    item.data.forEach(point => {
      if (point[0]) allTimestamps.push(point[0]);
    });
  });
  const minTime = allTimestamps.length > 0 ? Math.min(...allTimestamps) : undefined;
  const maxTime = allTimestamps.length > 0 ? Math.max(...allTimestamps) : undefined;

  return {
    title: {
      text: title,
      left: "center",
      top: 0,
      textStyle: {
        color: "#303133",
        fontSize: 14,
        fontWeight: 600
      }
    },
    backgroundColor: "transparent",
    animation: true,
    animationDuration: 750,
    animationEasing: "cubicOut",
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(255, 255, 255, 0.95)",
      borderColor: colors[0],
      borderWidth: 2,
      textStyle: {
        color: "#1f2937",
        fontSize: 12
      },
      axisPointer: {
        type: "line",
        lineStyle: {
          color: colors[0],
          width: 2,
          type: "dashed"
        }
      },
      formatter: (params: any) => {
        if (!params || !Array.isArray(params)) return "";
        const timeValue = params[0].axisValue;
        let timeStr = "";
        if (typeof timeValue === "number") {
          const date = new Date(timeValue);
          if (!isNaN(date.getTime())) {
            const pad = (n: number) => String(n).padStart(2, "0");
            timeStr = `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
          }
        }
        let html = `<div style="margin-bottom: 6px; font-weight: 600;">${timeStr}</div>`;
        params.forEach((item: any) => {
          if (item.value !== null && item.value !== undefined) {
            const value = Array.isArray(item.value) ? item.value[item.value.length - 1] : item.value;
            const color = item.color || colors[0];
            html += `<div style="margin-top: 4px; display: flex; align-items: center;">
              <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: ${color}; margin-right: 6px;"></span>
              <span>${item.seriesName || ""}: <strong style="color: ${color};">${value} ${unit}</strong></span>
            </div>`;
          }
        });
        return html;
      }
    },
    legend: {
      show: true,
      data: seriesData.map(s => s.name),
      bottom: 10,
      left: "center",
      icon: "line",
      itemWidth: 25,
      itemHeight: 14,
      itemGap: 20,
      textStyle: {
        fontSize: 12,
        padding: [0, 5, 0, 5]
      },
      backgroundColor: "rgba(255, 255, 255, 0.7)",
      borderRadius: 4,
      padding: [8, 12]
    },
    grid: {
      top: 50,
      right: 30,
      bottom: 100,
      left: 60,
      containLabel: true
    },
    xAxis: {
      type: "time",
      boundaryGap: false,
      min: minTime,
      max: maxTime,
      minInterval: 60 * 1000,
      maxInterval: 24 * 60 * 60 * 1000,
      axisLabel: {
        color: "#606266",
        fontSize: 12,
        rotate: -45,
        margin: 12,
        formatter: (value: number) => {
          if (!value) return "";
          const date = new Date(value);
          if (isNaN(date.getTime())) return "";
          const pad = (n: number) => String(n).padStart(2, "0");
          const month = date.getMonth() + 1;
          const day = date.getDate();
          const hours = date.getHours();
          const minutes = date.getMinutes();
          return `${month}/${day} ${pad(hours)}:${pad(minutes)}`;
        }
      },
      axisLine: {
        lineStyle: {
          color: "#E4E7ED",
          width: 1
        }
      },
      splitLine: {
        show: true,
        lineStyle: {
          color: "#EBEEF5",
          type: "dashed",
          width: 1
        }
      }
    },
    yAxis: {
      type: "value",
      name: unit,
      nameLocation: "end",
      nameTextStyle: {
        fontSize: 12,
        padding: [0, 0, 0, 10]
      },
      axisLabel: {
        fontSize: 12
      },
      axisLine: {
        lineStyle: {
          color: "#E4E7ED",
          width: 1
        }
      },
      splitLine: {
        show: true,
        lineStyle: {
          color: "#EBEEF5",
          type: "dashed",
          width: 1
        }
      }
    },
    series
  };
}

// 处理测缝计数据
function processCrackData(data: CrackDataPoint[]) {
  const seriesMap: Record<string, Array<[number, number | null]>> = {};
  
  CRACK_NUMBERS.forEach(num => {
    seriesMap[num] = [];
  });

  data.forEach(item => {
    if (item.number && CRACK_NUMBERS.includes(String(item.number))) {
      const timestamp = typeof item.timestamp === "string" 
        ? parseInt(item.timestamp) * 1000 
        : (item.timestamp as number) * 1000;
      const value = item.data1 !== null && item.data1 !== undefined ? item.data1 : null;
      seriesMap[String(item.number)].push([timestamp, value]);
    }
  });

  // 按时间戳排序
  Object.keys(seriesMap).forEach(key => {
    seriesMap[key].sort((a, b) => a[0] - b[0]);
  });

  return Object.keys(seriesMap).map(name => ({
    name,
    data: seriesMap[name]
  }));
}

// 处理测斜探头数据
function processTiltData(data: TiltDataPoint[], direction: "x" | "y") {
  const seriesMap: Record<string, Array<[number, number | null]>> = {};
  
  TILT_NUMBERS.forEach(num => {
    seriesMap[num] = [];
  });

  data.forEach(item => {
    if (item.number && TILT_NUMBERS.includes(String(item.number))) {
      const timestamp = typeof item.timestamp === "string" 
        ? parseInt(item.timestamp) * 1000 
        : (item.timestamp as number) * 1000;
      const value = direction === "x" 
        ? (item.data1 !== null && item.data1 !== undefined ? item.data1 : null)
        : (item.data2 !== null && item.data2 !== undefined ? item.data2 : null);
      seriesMap[String(item.number)].push([timestamp, value]);
    }
  });

  Object.keys(seriesMap).forEach(key => {
    seriesMap[key].sort((a, b) => a[0] - b[0]);
  });

  return Object.keys(seriesMap).map(name => ({
    name,
    data: seriesMap[name]
  }));
}

// 处理水准仪数据
function processLevelData(data: LevelDataPoint[]) {
  const seriesMap: Record<string, Array<[number, number | null]>> = {};
  
  LEVEL_NUMBERS.forEach(num => {
    seriesMap[num] = [];
  });

  data.forEach(item => {
    if (item.number && LEVEL_NUMBERS.includes(String(item.number))) {
      const timestamp = typeof item.timestamp === "string" 
        ? parseInt(item.timestamp) * 1000 
        : (item.timestamp as number) * 1000;
      const value = item.data1 !== null && item.data1 !== undefined ? item.data1 : null;
      seriesMap[String(item.number)].push([timestamp, value]);
    }
  });

  Object.keys(seriesMap).forEach(key => {
    seriesMap[key].sort((a, b) => a[0] - b[0]);
  });

  return Object.keys(seriesMap).map(name => ({
    name,
    data: seriesMap[name]
  }));
}

// 处理水位计数据
function processWaterLevelData(data: WaterLevelDataPoint[]) {
  const seriesMap: Record<string, Array<[number, number | null]>> = {};
  
  // 只显示number=478967的设备
  const TARGET_NUMBER = "478967";
  
  // 只保留目标设备的数据
  const filteredData = data.filter(item => {
    const itemNumber = String(item.number || "").replace(/\D/g, "").padStart(6, "0");
    return itemNumber === TARGET_NUMBER;
  });

  if (filteredData.length > 0) {
    seriesMap[TARGET_NUMBER] = [];
    
    filteredData.forEach(item => {
      const timestamp = typeof item.timestamp === "string" 
        ? parseInt(item.timestamp) * 1000 
        : (item.timestamp as number) * 1000;
      const value = item.data1 !== null && item.data1 !== undefined ? item.data1 : null;
      seriesMap[TARGET_NUMBER].push([timestamp, value]);
    });

    seriesMap[TARGET_NUMBER].sort((a, b) => a[0] - b[0]);
  }

  return Object.keys(seriesMap).map(name => ({
    name,
    data: seriesMap[name]
  }));
}

// 更新图表
async function updateCharts() {
  try {
    // 获取所有数据
    const [crackData, tiltData, levelData, waterLevelData] = await Promise.all([
      fetchCrackData(),
      fetchTiltData(),
      fetchLevelData(),
      fetchWaterLevelData()
    ]);

    // 更新测缝计图表
    if (crackChart && crackChartRef.value) {
      const crackSeries = processCrackData(crackData);
      const crackOption = createChartOption("Crack meter", crackSeries, "");
      crackChart.setOption(crackOption as any, { notMerge: true });
    }

    // 更新测斜探头-X图表
    if (tiltXChart && tiltXChartRef.value) {
      const tiltXSeries = processTiltData(tiltData, "x");
      const tiltXOption = createChartOption("Tilt probe (X)", tiltXSeries, "");
      tiltXChart.setOption(tiltXOption as any, { notMerge: true });
    }

    // 更新测斜探头-Y图表
    if (tiltYChart && tiltYChartRef.value) {
      const tiltYSeries = processTiltData(tiltData, "y");
      const tiltYOption = createChartOption("Tilt probe (Y)", tiltYSeries, "");
      tiltYChart.setOption(tiltYOption as any, { notMerge: true });
    }

    // 更新水准仪图表
    if (levelChart && levelChartRef.value) {
      const levelSeries = processLevelData(levelData);
      const levelOption = createChartOption("Settlement", levelSeries, "");
      levelChart.setOption(levelOption as any, { notMerge: true });
    }

    // 更新水位计图表
    if (waterLevelChart && waterLevelChartRef.value) {
      const waterLevelSeries = processWaterLevelData(waterLevelData);
      const waterLevelOption = createChartOption("Water level", waterLevelSeries, "mm");
      waterLevelChart.setOption(waterLevelOption as any, { notMerge: true });
    }

    // 更新最后更新时间
    lastUpdateTime.value = new Date().toLocaleString("zh-CN");
  } catch (error) {
    console.error("Failed to fetch data:", error);
  }
}

// 初始化图表
function initCharts() {
  if (crackChartRef.value) {
    crackChart = echarts.init(crackChartRef.value);
  }
  if (tiltXChartRef.value) {
    tiltXChart = echarts.init(tiltXChartRef.value);
  }
  if (tiltYChartRef.value) {
    tiltYChart = echarts.init(tiltYChartRef.value);
  }
  if (levelChartRef.value) {
    levelChart = echarts.init(levelChartRef.value);
  }
  if (waterLevelChartRef.value) {
    waterLevelChart = echarts.init(waterLevelChartRef.value);
  }

  // 监听窗口大小变化
  window.addEventListener("resize", handleResize);
}

// 处理窗口大小变化
function handleResize() {
  crackChart?.resize();
  tiltXChart?.resize();
  tiltYChart?.resize();
  levelChart?.resize();
  waterLevelChart?.resize();
}

// 手动刷新数据
function refreshData() {
  updateCharts();
}

// 启动定时刷新
function startAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
  refreshTimer = window.setInterval(() => {
    updateCharts();
  }, REFRESH_INTERVAL);
}

// 组件挂载
onMounted(() => {
  initCharts();
  updateCharts();
  startAutoRefresh();
});

// 组件卸载
onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
  window.removeEventListener("resize", handleResize);
  crackChart?.dispose();
  tiltXChart?.dispose();
  tiltYChart?.dispose();
  levelChart?.dispose();
  waterLevelChart?.dispose();
});
</script>

<style scoped>
.shm-monitor-container {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding: 15px 20px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.header h2 {
  margin: 0;
  font-size: 20px;
  color: #303133;
}

.refresh-info {
  display: flex;
  align-items: center;
  gap: 15px;
}

.refresh-info span {
  color: #606266;
  font-size: 14px;
}

.chart-container {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
}

.chart-container.full-width {
  width: 100%;
}

.chart-container.half-width {
  flex: 1;
  margin: 0 10px;
}

.chart-row {
  display: flex;
  gap: 20px;
}

.chart-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 2px solid #409eff;
}

.chart {
  width: 100%;
  height: 400px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .chart-row {
    flex-direction: column;
  }
  
  .chart-container.half-width {
    margin: 0;
    margin-bottom: 20px;
  }
}
</style>

