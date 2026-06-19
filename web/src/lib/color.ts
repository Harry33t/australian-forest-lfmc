// 暖色调配色。植被色 + LFMC/不确定性连续色阶(贴野火主题)。
export const VEG_COLOR: Record<string, [number, number, number]> = {
  forest: [104, 122, 58],     // 暖橄榄绿
  shrubland: [190, 95, 45],   // 赭土 / terracotta
  grassland: [214, 166, 74],  // 麦金
};
export const VEG_LABEL: Record<string, string> = {
  forest: "Forest",
  shrubland: "Shrubland",
  grassland: "Grassland",
};

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function ramp(stops: [number, [number, number, number]][], v: number): [number, number, number] {
  const c = Math.max(stops[0][0], Math.min(stops[stops.length - 1][0], v));
  for (let i = 0; i < stops.length - 1; i++) {
    const [x0, c0] = stops[i], [x1, c1] = stops[i + 1];
    if (c >= x0 && c <= x1) {
      const t = (c - x0) / (x1 - x0 || 1);
      return [lerp(c0[0], c1[0], t), lerp(c0[1], c1[1], t), lerp(c0[2], c1[2], t)];
    }
  }
  return stops[stops.length - 1][1];
}

// LFMC:暖色阶,干(低)深褐→赭→琥珀→麦→奶油(高,湿)。整体暖、贴野火。
const LFMC_STOPS: [number, [number, number, number]][] = [
  [40, [92, 38, 24]],
  [80, [168, 72, 34]],
  [110, [212, 130, 48]],
  [160, [236, 184, 96]],
  [220, [246, 224, 168]],
];
export const lfmcColor = (v: number) => ramp(LFMC_STOPS, v).map(Math.round) as [number, number, number];

// 不确定性:低(稳)暖褐 → 高(不稳)赤红(警示)
const UNC_STOPS: [number, [number, number, number]][] = [
  [2, [150, 116, 70]],
  [12, [210, 140, 60]],
  [30, [206, 80, 46]],
  [50, [168, 40, 34]],
];
export const uncColor = (v: number) => ramp(UNC_STOPS, v).map(Math.round) as [number, number, number];

// 迁移矩阵 R²:负(赤红)→0(浅米中性,配浅色卡)→正(暖橄榄绿)
export function r2Color(v: number): string {
  const c = Math.max(-1, Math.min(1, v));
  const neutral = [240, 233, 222];
  if (c >= 0) {
    const t = c, g = [120, 146, 64];
    return `rgb(${Math.round(lerp(neutral[0], g[0], t))},${Math.round(lerp(neutral[1], g[1], t))},${Math.round(lerp(neutral[2], g[2], t))})`;
  }
  const t = -c, r = [186, 58, 42];
  return `rgb(${Math.round(lerp(neutral[0], r[0], t))},${Math.round(lerp(neutral[1], r[1], t))},${Math.round(lerp(neutral[2], r[2], t))})`;
}
