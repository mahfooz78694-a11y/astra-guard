import sys, os, unittest, torch, torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from astra_guard import ZVILGuard, AutoSubspaceTuner
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Conv2d(3, 16, 3, padding=1)
    def forward(self, x): return self.layer4(x)
class TestAstra(unittest.TestCase):
    def test_pipeline(self):
        m = DummyModel().eval()
        g = ZVILGuard(m, 'layer4', rank_k=4)
        g.calibrate([torch.randn(4, 3, 8, 8)])
        out = m(torch.randn(4, 3, 8, 8))
        self.assertEqual(out.shape, torch.Size([4, 16, 8, 8]))
if __name__ == '__main__': unittest.main()