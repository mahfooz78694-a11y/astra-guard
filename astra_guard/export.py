import torch
def export_protected_onnx(model, dummy_input, export_path):
    torch.onnx.export(model, dummy_input, export_path, opset_version=17)
