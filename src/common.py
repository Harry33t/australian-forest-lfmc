"""F2 共享常量 / 工具(卫星 LFMC 侧)。

Globe-LFMC 2.0 字段均核自数据集与论文(Yebra et al. 2024, Sci Data 11:332,
DOI 10.1038/s41597-024-03159-6;figshare 25413790):
  - 单个 .xlsx,3 个 sheet:`LFMC Data`(主表,一行一次测量) / `Contact` / `Protocol`
  - 目标变量:`LFMC value (%)` = 干重百分比 (Wf-Wd)/Wd*100,可 >100%,不是 0–1 分数
  - 植被分型:`Species functional type`(Tree/Shrub/Grass + small tree/large shrub 等中间型)
  - 土地覆盖:`IGBP Land Cover`(已预 join MODIS MCD12Q1,无需外接栅格)
  - 坐标:`Latitude (WGS84, EPSG:4326)` / `Longitude (WGS84, EPSG:4326)`
  - 日期:`Sampling date (YYYYMMDD)`,整数 YYYYMMDD(非 Excel/ISO 日期),需显式 parse
  - 自带气象列:多窗口降水(24h/3d/1w/4w/12w)、4 次/日 RH、气温、水汽压、风速

对标基线(诚实版):Yebra RS 2026 (18(7):1049) 报的森林 R²=0.43/灌丛 0.21/草地 0.83
是 AFMS/MODIS *仿真器*(卫星→卫星);她对 Globe-LFMC 2.0 *地面真值* 的验证是
R²≈0.42(同质站点)~0.53(定制外业)。我们做地面 LFMC 直接检索 → 对标 0.42–0.53。
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 数据集来源
# ---------------------------------------------------------------------------
FIGSHARE_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/45049786"
XLSX_NAME = "Globe-LFMC-2.0 final.xlsx"
SHEET_DATA = "LFMC data"           # 主表(注意:小写 d,核自文件)
SHEET_CONTACT = "Contact"
SHEET_PROTOCOL = "Protocol"

# ---------------------------------------------------------------------------
# 主表列名(带空格/括号/单位,引用必须逐字)——在 prepare 阶段会 rename 成短名
# ---------------------------------------------------------------------------
COL_SITE = "Site name"
COL_COUNTRY = "Country"
COL_STATE = "State/Region"
COL_LAT = "Latitude (WGS84, EPSG:4326)"
COL_LON = "Longitude (WGS84, EPSG:4326)"
COL_DATE = "Sampling date (YYYYMMDD)"
COL_LFMC = "LFMC value (%)"
COL_FUNCTYPE = "Species functional type"
COL_SPECIES = "Species collected"
COL_IGBP = "IGBP Land Cover"
COL_ELEV = "Elevation (m.a.s.l)"
COL_SLOPE = "Slope (%)"

# 自带气象列(核自文件第 24–37 列):LFMC 的强物理驱动,零成本,W1 就纳入
COL_MET = {
    "Precipitation 24h sum (mm/day)": "precip_24h",
    "Precipitation sum 3 days before (mm/day)": "precip_3d",
    "Precipitation sum 1 week before (mm/day)": "precip_1w",
    "Precipitation sum 4 weeks before (mm/day)": "precip_4w",
    "Precipitation sum 12 weeks before (mm/day)": "precip_12w",
    "2m Relative Humidity at 06h (%)": "rh_06",
    "2m Relative Humidity at 09h (%)": "rh_09",
    "2m Relative Humidity at 12h (%)": "rh_12",
    "2m Relative Humidity at 15h (%)": "rh_15",
    "2m Air Temperature 24h max (K)": "temp_max",
    "2m Air Temperature 24h mean (K)": "temp_mean",
    "Vapour Pressure 24h mean (hPa)": "vp_mean",
    "10m Wind Speed 24h mean (m/s)": "wind_mean",
    "2m Dewpoint Temperature 24h mean (K)": "dewpoint_mean",
}

# 短名(prepare 后用这套,代码里到处引用)
RENAME = {
    COL_SITE: "site",
    COL_COUNTRY: "country",
    COL_STATE: "state",
    COL_LAT: "lat",
    COL_LON: "lon",
    COL_DATE: "date",
    COL_LFMC: "lfmc",
    COL_FUNCTYPE: "func_type",
    COL_SPECIES: "species",
    COL_IGBP: "igbp",
    COL_ELEV: "elevation",
    COL_SLOPE: "slope",
    **COL_MET,
}

# ---------------------------------------------------------------------------
# 植被三分类:对标 Yebra 分层报 R² 的 forest / shrubland / grassland
# 优先用 `Species functional type`(逐样本植物功能型);中间型按主导归并。
# ---------------------------------------------------------------------------
VEG_FOREST, VEG_SHRUB, VEG_GRASS = "forest", "shrubland", "grassland"
VEG_TYPES = [VEG_FOREST, VEG_SHRUB, VEG_GRASS]

# func_type 原值(小写、去空格后)→ 三分类;未列出的归 None(建模时丢弃)
FUNCTYPE_TO_VEG = {
    "tree": VEG_FOREST,
    "smalltree": VEG_FOREST,
    "shrub": VEG_SHRUB,
    "largeshrub": VEG_SHRUB,
    "smallshrub": VEG_SHRUB,
    "grass": VEG_GRASS,
    "forb": VEG_GRASS,        # 草本归草地一类(动态范围相近)
}

# 兜底:func_type 缺失时用 IGBP 折叠(关键词匹配,全小写)
IGBP_FOREST_KW = ["forest"]
IGBP_SHRUB_KW = ["shrubland", "shrub"]
IGBP_GRASS_KW = ["grassland", "savanna"]   # savanna(含 woody savanna)归草地侧

# ---------------------------------------------------------------------------
# 站点/气象侧特征(W1 零卫星 baseline 用;卫星光谱特征 W2 接 DEA 后再加)
# 这些是 Globe-LFMC 2.0 自带列的短名,prepare 阶段抽出能用的数值列。
# ---------------------------------------------------------------------------
SITE_FEATURES = ["lat", "lon", "elevation", "slope"]
# 时间特征(从 date 派生):年内日序的正余弦编码,捕捉季节性
TIME_FEATURES = ["doy_sin", "doy_cos"]
# 气象特征(自带列短名):LFMC 强物理驱动,W1 baseline 主力
MET_FEATURES = list(COL_MET.values())

# 卫星光谱特征(satellite_dea.py 产出):7 个 S2 反射率 + 6 个 LFMC 指数
S2_REFL = ["s2_blue", "s2_green", "s2_red", "s2_nir", "s2_nir_narrow", "s2_swir1", "s2_swir2"]
S2_INDICES = ["ndvi", "ndii", "ndwi", "gvmi", "nmdi", "vari"]
S2_FEATURES = S2_REFL + S2_INDICES

# 特征集预设(baseline_rf.py --feature-set 用):met=W1 地板,s2=纯卫星,met+s2=全量
FEATURE_SETS = {
    "base": SITE_FEATURES + TIME_FEATURES,
    "met": SITE_FEATURES + TIME_FEATURES + MET_FEATURES,
    "s2": SITE_FEATURES + TIME_FEATURES + S2_FEATURES,
    "met+s2": SITE_FEATURES + TIME_FEATURES + MET_FEATURES + S2_FEATURES,
}

# ---------------------------------------------------------------------------
# 卫星光谱指数公式(W2 接 DEA Sentinel-2 后用;此处先记公式,核自 LFMC 文献)
# Sentinel-2 波段:B2 蓝 B3 绿 B4 红 B8 NIR B8A NIR-narrow B11 SWIR1 B12 SWIR2
#   NDVI = (B8-B4)/(B8+B4)
#   NDII/NDMI = (B8-B11)/(B8+B11)        ← LFMC 文献中最强预测因子
#   NDWI(Gao) = (B8-B11)/(B8+B11);SWIR2 变体 (B8-B12)/(B8+B12)
#   GVMI = ((B8A+0.1)-(B12+0.02)) / ((B8A+0.1)+(B12+0.02))
#   NMDI = (B8A-(B11-B12)) / (B8A+(B11-B12))
#   VARI = (B3-B4)/(B3+B4-B2)
# Yebra RS 2026 基线用:10 个 S2 波段 + NDVI + NDII,RF 回归,无气象变量。
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 默认路径(全部可被 CLI 覆盖;本地/AutoDL 一套代码)
#   data/raw/    大原始,不进 git、不传 AutoDL
#   outputs/     小产物 parquet/特征表/结果,这才是要 rsync 的
# ---------------------------------------------------------------------------
RAW_DIR = "data/raw"
PROCESSED_PARQUET = "outputs/lfmc_clean.parquet"      # 全球清洗后
AU_PARQUET = "outputs/lfmc_au.parquet"                # 澳洲子集
RESULTS_DIR = "outputs/results"

RANDOM_SEED = 0


def normalize_functype(s):
    """func_type 原始字符串 → 规范 key(小写、去空格/连字符)。"""
    if s is None:
        return ""
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def map_veg_type(func_type, igbp=None):
    """逐样本三分类:先用 func_type,缺失/未知再用 IGBP 兜底;都不行返回 None。"""
    key = normalize_functype(func_type)
    if key in FUNCTYPE_TO_VEG:
        return FUNCTYPE_TO_VEG[key]
    if igbp:
        ig = str(igbp).strip().lower()
        if any(k in ig for k in IGBP_FOREST_KW):
            return VEG_FOREST
        if any(k in ig for k in IGBP_SHRUB_KW):
            return VEG_SHRUB
        if any(k in ig for k in IGBP_GRASS_KW):
            return VEG_GRASS
    return None
