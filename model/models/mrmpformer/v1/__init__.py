# MRMPFormer v1：三层 Decoder + FDR 边界逐层精化
# backbone 与 QuanFormer 相同（ResNet-50），Transformer 改为
# 3 层 Decoder + FDR 左右边界分布残差精化 + 逐层边界位置反馈。
# 实现见 detr.py / transformer.py / fdr.py，入口 build(args)。
