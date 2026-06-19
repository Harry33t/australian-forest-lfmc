export interface Site {
  site: string;
  lon: number;
  lat: number;
  veg: "forest" | "shrubland" | "grassland";
  bioregion: string;
  lfmc: number;   // 实测 LFMC 均值 (%)
  pred: number;   // 留站点交叉验证预测 (%)
  unc: number;    // 模型不确定性(RF 树间标准差)(%)
  n: number;      // 测量数
}

export interface Loro {
  feature_set: string;
  regions: string[];
  loro: Record<string, { r2: number; rmse: number; n: number }>;
  transfer_matrix: Record<string, Record<string, number | null>>;
}

export interface Conformal {
  feature_set: string;
  levels: number[];
  overall: Record<string, { coverage: number; width: number }>;
  r2: number;
  bands_example: { pred: number[]; lo: number[]; hi: number[]; true: number[] };
}

export interface VegStat {
  n: number; sites: number; median: number;
  q1: number; q3: number; p05: number; p95: number;
}
export interface Summary {
  n_measurements: number; n_sites: number; n_bioregions: number;
  year_min: number; year_max: number;
  veg_stats: Record<string, VegStat>;
  headline: Record<string, number>;
}

export type ColorMode = "lfmc" | "unc" | "veg";
