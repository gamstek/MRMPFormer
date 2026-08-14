# 模型家族根：build_model 工厂按 --model 参数路由到具体变体
from .quanformer.detr import build as build_quanformer


def build_model(args):
    """根据 args.model 选择模型变体。

    支持的变体：
      - quanformer      : QuanFormer baseline（默认）
      - mrmpformer_v1   : MRMPFormer v1（backbone 不变，Transformer 改）
    """
    variant = getattr(args, "model", "quanformer")
    if variant == "quanformer":
        return build_quanformer(args)
    elif variant == "mrmpformer_v1":
        from .mrmpformer.v1.detr import build as build_mrmpformer_v1
        return build_mrmpformer_v1(args)
    else:
        raise ValueError(f"Unknown model variant: {variant}")
