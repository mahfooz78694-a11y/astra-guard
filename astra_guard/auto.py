import torch.nn as nn
class AutoSubspaceTuner:
    def __init__(self, energy_threshold: float = 0.9999): self.energy_threshold = energy_threshold
    def discover_optimal_layer(self, model: nn.Module) -> str:
        candidates = [(n, sum(p.numel() for p in m.parameters())) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
