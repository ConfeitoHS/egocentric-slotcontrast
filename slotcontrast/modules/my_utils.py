import torch
from torch import nn
from collections import OrderedDict

class HookedFeatureExtractor(nn.Module):
    """
    Wrap a model and return selected intermediate activations as a dict
    {module_name: activation}. It registers forward hooks on exact module names.
    """
    def __init__(self, model: nn.Module, target_names):  #dino_s_vitb14, ['blocks.11', 'blocks.11.attn.qkv']
        super().__init__()
        self.model = model
        self.target_names = list(target_names)
        self._features = OrderedDict()
        self._handles = []
        self._register_hooks()

    def _clear(self):
        self._features.clear()

    def _save(self, name):
        def fn(module, inp, out):
            # 不做 .detach()，保持梯度（若只做推理可改为 out.detach() 节约显存）
            self._features[name] = out
        return fn

    def _register_hooks(self):
        # 收集所有可选模块名
        name_to_module = dict(self.model.named_modules())  # '' -> whole model 也会在里头
        available = set(name_to_module.keys()) - {''}

        # 校验 + 挂钩
        for name in self.target_names:
            if name not in available:
                raise ValueError(
                    f"Requested feature node '{name}' not found. "
                    f"Available nodes (partial): {sorted(list(available))[:50]} ..."
                )
            h = name_to_module[name].register_forward_hook(self._save(name))
            self._handles.append(h)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def forward(self, x, *args, **kwargs):
        self._clear()
        _ = self.model(x, *args, **kwargs)   # 正常前向；hook 会把中间特征塞进 self._features
        # 返回一个常规 dict（下游和 torchvision 的 extractor 对齐）
        return dict(self._features)

    def __del__(self):
        self.remove_hooks()


class VitBlock12NoCLS(nn.Module):
    def __init__(self, model, HookedFeatureExtractor, features):
        super().__init__()
        self.model = model
        self.hfe = HookedFeatureExtractor(model, features)

    @torch.no_grad()
    def forward(self, x):
        outs = self.hfe(x)
        tokens = outs['blocks.11']              # [B, N, C]
        qkv    = outs['blocks.11.attn.qkv']     # [B, N, 3*C]

        return {
            'blocks.11.attn.qkv': qkv[:,1:,:],
            'blocks.11': tokens[:,1:,:]
        }

        # # ---- 估计前缀 token 数（CLS / DIST）并去掉 ----
        # # 优先用 grid_size 推出 patch 数；否则退化为不裁剪
        # num_patches = None
        # gs = getattr(self.model.patch_embed, 'grid_size', None)
        # if gs is not None and isinstance(gs, (tuple, list)) and len(gs) == 2:
        #     num_patches = int(gs[0]) * int(gs[1])

        # B, N, C = tokens.shape
        # if num_patches is not None and N >= num_patches:
        #     num_prefix = N - num_patches   # 常见为 1（CLS）或 2（CLS+DIST）
        #     if num_prefix > 0:
        #         tokens = tokens[:, num_prefix:, :]     # -> [B, 256, C]
        #         qkv    = qkv[:,    num_prefix:, :]     # -> [B, 256, 3*C]

        # # ---- 拆 q/k/v 并整理到 [B, H, 256, D] ----
        # # head 数：优先从模型读；否则默认 12
        # num_heads = getattr(self.model, 'num_heads', 12)
        # head_dim  = C // num_heads

        # qkv = qkv.view(B, tokens.shape[1], 3, num_heads, head_dim)  # [B, 256, 3, H, D]
        # q, k, v = qkv.unbind(dim=2)                                  # 各 [B, 256, H, D]
        # # 统一排成 [B, H, 256, D]
        # q = q.permute(0, 2, 1, 3).contiguous()
        # k = k.permute(0, 2, 1, 3).contiguous()
        # v = v.permute(0, 2, 1, 3).contiguous()

        # return {'tokens': tokens, 'q': q, 'k': k, 'v': v}
