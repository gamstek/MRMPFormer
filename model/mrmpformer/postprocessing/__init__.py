"""后处理模块：峰积分定量、质量评估、Box↔RT映射。"""
from mrmpformer.postprocessing.quantify import max_consecutive, AREA_TIME_UNIT_SCALE
from mrmpformer.postprocessing.mapping import box_to_rt_range
from mrmpformer.postprocessing.quality import compute_roi_quality_params
