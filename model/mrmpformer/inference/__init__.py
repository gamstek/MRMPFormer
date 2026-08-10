"""推理模块：模型加载、预测、可视化。"""
from mrmpformer.inference.predictor import build_predictor, predict, plot_results
from mrmpformer.inference.device import resolve_torch_device, load_torch_checkpoint
