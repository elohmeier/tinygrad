import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad import dtypes

def _run_onnx(model_proto, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
  import onnxruntime as ort
  sess = ort.InferenceSession(model_proto.SerializeToString())
  return sess.run(None, inputs)

def _export_and_compare(model, *inputs, rtol=1e-5, atol=1e-5):
  from extra.export_onnx import export_onnx
  # realize inputs
  for inp in inputs: inp.realize()
  # run in tinygrad
  out = model.forward(*inputs) if hasattr(model, "forward") else model(*inputs)
  expected = [out.numpy()] if isinstance(out, Tensor) else [o.numpy() for o in out]
  # export
  onnx_model = export_onnx(model, *inputs)
  # run in onnxruntime
  ort_inputs = {f"input{i}": inp.numpy() for i, inp in enumerate(inputs)}
  ort_out = _run_onnx(onnx_model, ort_inputs)
  # compare
  assert len(ort_out) == len(expected), f"output count mismatch: {len(ort_out)} vs {len(expected)}"
  for i, (actual, exp) in enumerate(zip(ort_out, expected)):
    np.testing.assert_allclose(actual, exp, rtol=rtol, atol=atol, err_msg=f"output {i} mismatch")

class TestExportOnnx(unittest.TestCase):
  def test_add_scalar(self):
    _export_and_compare(lambda x: x + 2.0, Tensor.rand(2, 3))

  def test_mul_scalar(self):
    _export_and_compare(lambda x: x * 3.0, Tensor.rand(2, 3))

  def test_binary_add(self):
    _export_and_compare(lambda x, y: x + y, Tensor.rand(2, 3), Tensor.rand(2, 3))

  def test_binary_mul(self):
    _export_and_compare(lambda x, y: x * y, Tensor.rand(2, 3), Tensor.rand(2, 3))

  def test_binary_sub(self):
    _export_and_compare(lambda x, y: x - y, Tensor.rand(2, 3), Tensor.rand(2, 3))

  def test_binary_div(self):
    _export_and_compare(lambda x, y: x / y, Tensor.rand(2, 3), Tensor.rand(2, 3) + 0.1)

  def test_neg(self):
    _export_and_compare(lambda x: -x, Tensor.rand(2, 3))

  def test_sqrt(self):
    _export_and_compare(lambda x: x.sqrt(), Tensor.rand(2, 3) + 0.01)

  def test_sin(self):
    _export_and_compare(lambda x: x.sin(), Tensor.rand(2, 3))

  def test_reciprocal(self):
    _export_and_compare(lambda x: x.reciprocal(), Tensor.rand(2, 3) + 0.1)

  def test_exp2(self):
    _export_and_compare(lambda x: x.exp2(), Tensor.rand(2, 3), rtol=1e-4, atol=1e-4)

  def test_log2(self):
    _export_and_compare(lambda x: x.log2(), Tensor.rand(2, 3) + 0.01, rtol=1e-4, atol=1e-4)

  def test_relu(self):
    _export_and_compare(lambda x: x.relu(), Tensor.rand(2, 3) - 0.5)

  def test_reshape(self):
    _export_and_compare(lambda x: x.reshape(3, 2), Tensor.rand(2, 3))

  def test_permute(self):
    _export_and_compare(lambda x: x.permute(1, 0), Tensor.rand(2, 3))

  def test_expand(self):
    _export_and_compare(lambda x: x.reshape(1, 6).expand(3, 6), Tensor.rand(2, 3))

  def test_pad(self):
    _export_and_compare(lambda x: x.pad(((1, 1), (2, 2))), Tensor.rand(2, 3))

  def test_shrink(self):
    _export_and_compare(lambda x: x[:, 1:3], Tensor.rand(4, 5))

  def test_sum(self):
    _export_and_compare(lambda x: x.sum(axis=1), Tensor.rand(2, 3))

  def test_sum_keepdim(self):
    _export_and_compare(lambda x: x.sum(axis=1, keepdim=True), Tensor.rand(2, 3))

  def test_max_reduce(self):
    _export_and_compare(lambda x: x.max(axis=0), Tensor.rand(3, 4))

  def test_mean(self):
    _export_and_compare(lambda x: x.mean(axis=-1), Tensor.rand(2, 3))

  def test_cast_float16(self):
    _export_and_compare(lambda x: x.half(), Tensor.rand(2, 3), rtol=1e-2, atol=1e-2)

  def test_cast_int32(self):
    _export_and_compare(lambda x: x.cast(dtypes.int32), Tensor.rand(2, 3) * 100)

  def test_where(self):
    _export_and_compare(lambda x: (x > 0.5).where(x, Tensor.zeros(2, 3)), Tensor.rand(2, 3))

  def test_linear_layer(self):
    class Linear:
      def __init__(self):
        self.weight = Tensor.rand(4, 3)
        self.bias = Tensor.rand(4)
      def __call__(self, x): return x @ self.weight.T + self.bias
    model = Linear()
    model.weight.realize()
    model.bias.realize()
    _export_and_compare(model, Tensor.rand(2, 3), rtol=1e-4, atol=1e-4)

  def test_two_layer_relu(self):
    class TwoLayer:
      def __init__(self):
        self.w1 = Tensor.rand(4, 3)
        self.b1 = Tensor.rand(4)
        self.w2 = Tensor.rand(2, 4)
        self.b2 = Tensor.rand(2)
      def __call__(self, x):
        x = (x @ self.w1.T + self.b1).relu()
        return x @ self.w2.T + self.b2
    model = TwoLayer()
    for t in [model.w1, model.b1, model.w2, model.b2]: t.realize()
    _export_and_compare(model, Tensor.rand(2, 3), rtol=1e-4, atol=1e-4)

  def test_multi_output(self):
    _export_and_compare(lambda x: (x + 1.0, x * 2.0), Tensor.rand(2, 3))

  def test_multi_input(self):
    _export_and_compare(lambda x, y, z: x + y + z, Tensor.rand(2, 3), Tensor.rand(2, 3), Tensor.rand(2, 3))

  def test_flip(self):
    _export_and_compare(lambda x: x.flip(0), Tensor.rand(3, 4))

  def test_sigmoid(self):
    _export_and_compare(lambda x: x.sigmoid(), Tensor.rand(2, 3))

  def test_tanh(self):
    _export_and_compare(lambda x: x.tanh(), Tensor.rand(2, 3))

  def test_softmax(self):
    _export_and_compare(lambda x: x.softmax(axis=-1), Tensor.rand(2, 3), rtol=1e-4, atol=1e-4)

  def test_matmul(self):
    _export_and_compare(lambda x, y: x @ y, Tensor.rand(2, 3), Tensor.rand(3, 4), rtol=1e-4, atol=1e-4)

  # --- conv2d tests ---

  def test_conv2d_no_pad(self):
    w = Tensor.rand(16, 3, 3, 3).realize()
    _export_and_compare(lambda x: x.conv2d(w), Tensor.rand(1, 3, 8, 8), rtol=1e-4, atol=1e-4)

  def test_conv2d_with_pad(self):
    w = Tensor.rand(16, 3, 3, 3).realize()
    _export_and_compare(lambda x: x.conv2d(w, padding=1), Tensor.rand(1, 3, 8, 8), rtol=1e-4, atol=1e-4)

  def test_conv2d_stride2(self):
    w = Tensor.rand(16, 3, 3, 3).realize()
    _export_and_compare(lambda x: x.conv2d(w, stride=2, padding=1), Tensor.rand(1, 3, 16, 16), rtol=1e-4, atol=1e-4)

  def test_conv2d_with_bias(self):
    w = Tensor.rand(16, 3, 3, 3).realize()
    b = Tensor.rand(16).realize()
    _export_and_compare(lambda x: x.conv2d(w, b, padding=1), Tensor.rand(1, 3, 8, 8), rtol=1e-4, atol=1e-4)

  def test_conv2d_depthwise(self):
    w = Tensor.rand(6, 1, 3, 3).realize()
    _export_and_compare(lambda x: x.conv2d(w, groups=6), Tensor.rand(1, 6, 8, 8), rtol=1e-4, atol=1e-4)

  def test_conv2d_grouped(self):
    w = Tensor.rand(12, 2, 3, 3).realize()
    _export_and_compare(lambda x: x.conv2d(w, groups=3), Tensor.rand(1, 6, 8, 8), rtol=1e-4, atol=1e-4)

  def test_small_cnn(self):
    class SmallCNN:
      def __init__(self):
        self.conv1 = Tensor.rand(16, 3, 3, 3).realize()
        self.conv2 = Tensor.rand(32, 16, 3, 3).realize()
        self.fc = Tensor.rand(10, 32).realize()
        self.fc_bias = Tensor.rand(10).realize()
      def __call__(self, x):
        x = x.conv2d(self.conv1, padding=1).relu()
        x = x.conv2d(self.conv2, padding=1).relu()
        x = x.mean(axis=(2, 3))  # global average pool
        return x @ self.fc.T + self.fc_bias
    model = SmallCNN()
    _export_and_compare(model, Tensor.rand(2, 3, 8, 8), rtol=1e-3, atol=1e-3)

  # --- pooling tests ---

  def test_max_pool2d(self):
    _export_and_compare(lambda x: x.max_pool2d((2, 2)), Tensor.rand(1, 3, 8, 8))

  def test_avg_pool2d(self):
    _export_and_compare(lambda x: x.avg_pool2d((2, 2)), Tensor.rand(1, 3, 8, 8), rtol=1e-4, atol=1e-4)

  def test_max_pool2d_stride(self):
    _export_and_compare(lambda x: x.max_pool2d((2, 2), stride=2), Tensor.rand(1, 16, 8, 8))

  # --- pattern node verification ---

  def test_relu_emits_relu_node(self):
    from extra.export_onnx import export_onnx
    x = Tensor.rand(2, 3).realize()
    m = export_onnx(lambda x: x.relu(), x)
    ops = [n.op_type for n in m.graph.node]
    self.assertIn("Relu", ops)

  def test_sigmoid_emits_sigmoid_node(self):
    from extra.export_onnx import export_onnx
    x = Tensor.rand(2, 3).realize()
    m = export_onnx(lambda x: x.sigmoid(), x)
    ops = [n.op_type for n in m.graph.node]
    self.assertIn("Sigmoid", ops)

  def test_softmax_emits_softmax_node(self):
    from extra.export_onnx import export_onnx
    x = Tensor.rand(2, 3).realize()
    m = export_onnx(lambda x: x.softmax(axis=-1), x)
    ops = [n.op_type for n in m.graph.node]
    self.assertIn("Softmax", ops)

  def test_cnn_emits_conv_relu_nodes(self):
    from extra.export_onnx import export_onnx
    from collections import Counter
    w1 = Tensor.rand(16, 3, 3, 3).realize()
    w2 = Tensor.rand(32, 16, 3, 3).realize()
    fc = Tensor.rand(10, 32).realize()
    fc_b = Tensor.rand(10).realize()
    def model(x):
      x = x.conv2d(w1, padding=1).relu()
      x = x.conv2d(w2, padding=1).relu()
      x = x.mean(axis=(2, 3))
      return x @ fc.T + fc_b
    x = Tensor.rand(2, 3, 8, 8).realize()
    m = export_onnx(model, x)
    ops = Counter(n.op_type for n in m.graph.node)
    self.assertEqual(ops["Conv"], 2)
    self.assertEqual(ops["Relu"], 2)
    self.assertEqual(ops["Gemm"], 1)

  def test_resnet_block(self):
    """Test a ResNet-like block: conv+relu+conv+residual add."""
    w1 = Tensor.rand(16, 16, 3, 3).realize()
    w2 = Tensor.rand(16, 16, 3, 3).realize()
    def block(x):
      residual = x
      x = x.conv2d(w1, padding=1).relu()
      x = x.conv2d(w2, padding=1)
      return (x + residual).relu()
    _export_and_compare(block, Tensor.rand(1, 16, 8, 8), rtol=1e-3, atol=1e-3)

  # --- gelu ---

  def test_gelu(self):
    _export_and_compare(lambda x: x.gelu(), Tensor.rand(2, 3), rtol=1e-4, atol=1e-4)

  def test_gelu_emits_gelu_node(self):
    from extra.export_onnx import export_onnx
    m = export_onnx(lambda x: x.gelu(), Tensor.rand(2, 3).realize())
    self.assertIn("Gelu", [n.op_type for n in m.graph.node])

  # --- interpolate ---

  def test_interpolate_nearest(self):
    _export_and_compare(lambda x: x.interpolate((8, 8), mode='nearest'), Tensor.rand(1, 3, 4, 4))

  def test_interpolate_linear(self):
    _export_and_compare(lambda x: x.interpolate((8, 8), mode='linear'), Tensor.rand(1, 3, 4, 4), rtol=1e-3, atol=1e-3)

  def test_interpolate_emits_resize(self):
    from extra.export_onnx import export_onnx
    m = export_onnx(lambda x: x.interpolate((8, 8), mode='nearest'), Tensor.rand(1, 3, 4, 4).realize())
    self.assertIn("Resize", [n.op_type for n in m.graph.node])

  # --- conv_transpose2d ---

  def test_conv_transpose2d(self):
    w = Tensor.rand(16, 8, 3, 3).realize()
    _export_and_compare(lambda x: x.conv_transpose2d(w), Tensor.rand(1, 16, 4, 4), rtol=1e-3, atol=1e-3)

  def test_conv_transpose2d_stride2(self):
    w = Tensor.rand(64, 32, 2, 2).realize()
    _export_and_compare(lambda x: x.conv_transpose2d(w, stride=2), Tensor.rand(1, 64, 8, 8), rtol=1e-3, atol=1e-3)

  # --- FPN-like model ---

  def test_fpn_like(self):
    """FPN-like block: conv + relu + interpolate + conv (as in DBNet)."""
    w1 = Tensor.rand(32, 16, 1, 1).realize()
    w2 = Tensor.rand(16, 32, 3, 3).realize()
    def fpn(x):
      x = x.conv2d(w1).relu()
      x = x.interpolate((16, 16), mode='nearest')
      return x.conv2d(w2, padding=1).relu()
    _export_and_compare(fpn, Tensor.rand(1, 16, 8, 8), rtol=1e-3, atol=1e-3)

  def test_lstm_cell(self):
    from tinygrad import nn
    lstm = nn.LSTMCell(16, 32)
    for p in nn.state.get_state_dict(lstm).values(): p.realize()
    h = Tensor.zeros(2, 32).realize()
    c = Tensor.zeros(2, 32).realize()
    _export_and_compare(lambda x: lstm(x, (h, c))[0], Tensor.rand(2, 16), rtol=1e-4, atol=1e-4)

  def test_bilstm(self):
    """BiLSTM as used in CRNN: forward + backward LSTM, concat outputs."""
    from tinygrad import nn
    fwd = nn.LSTMCell(16, 32)
    bwd = nn.LSTMCell(16, 32)
    for p in nn.state.get_state_dict(fwd).values(): p.realize()
    for p in nn.state.get_state_dict(bwd).values(): p.realize()
    def bilstm(x):
      # x: (seq_len, batch, features) = (4, 2, 16)
      h_f, c_f = Tensor.zeros(2, 32), Tensor.zeros(2, 32)
      h_b, c_b = Tensor.zeros(2, 32), Tensor.zeros(2, 32)
      # forward pass: process seq left-to-right
      h_f, c_f = fwd(x[0], (h_f, c_f))
      h_f, c_f = fwd(x[1], (h_f, c_f))
      # backward pass: process seq right-to-left
      h_b, c_b = bwd(x[1], (h_b, c_b))
      h_b, c_b = bwd(x[0], (h_b, c_b))
      return h_f.cat(h_b, dim=-1)
    _export_and_compare(bilstm, Tensor.rand(2, 2, 16), rtol=1e-3, atol=1e-3)

class TestExportOnnxDETR(unittest.TestCase):
  """Test ONNX export of the DETR object detection model used in the TATR pipeline."""

  @classmethod
  def setUpClass(cls):
    import sys, os
    # Add cellgrab to path for DETR model access
    cellgrab_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'cellgrab')
    sys.path.insert(0, os.path.abspath(cellgrab_path))

  def _build_detr(self, num_queries=15, num_classes=2):
    from cellgrab_core.ml.tatr_config import TatrConfig
    from cellgrab_core.ml.detr.models.detr import build
    from tinygrad.nn.state import get_state_dict, load_state_dict

    config = TatrConfig(num_queries=num_queries, num_classes=num_classes, dropout=0.0)
    model = build(config)
    # Realize weights by round-tripping through numpy (tinygrad deepcopy quirk)
    sd = get_state_dict(model)
    realized_sd = {k: Tensor(v.numpy()) for k, v in sd.items()}
    load_state_dict(model, realized_sd)
    return model

  def _make_wrapper(self, model):
    from cellgrab_core.ml.detr.util.misc import NestedTensor
    class DETRExportWrapper:
      def __init__(self, detr):
        self.detr = detr
      def forward(self, x, mask):
        samples = NestedTensor(x, mask)
        features, pos = self.detr.backbone(samples)
        src, mask2 = features[-1].decompose()
        hs = self.detr.transformer(self.detr.input_proj(src), mask2, self.detr.query_embed.weight, pos[-1])[0]
        pred_logits = self.detr.class_embed(hs)
        pred_boxes = self.detr.bbox_embed(hs).sigmoid()
        return pred_logits[-1], pred_boxes[-1]
    return DETRExportWrapper(model)

  def test_detr_detection_export(self):
    """Export DETR detection model (ResNet18 + 6-layer transformer, 15 queries, 2 classes)."""
    model = self._build_detr(num_queries=15, num_classes=2)
    wrapper = self._make_wrapper(model)
    inp = Tensor(np.random.rand(1, 3, 128, 128).astype(np.float32)).realize()
    mask = Tensor(np.zeros((1, 128, 128), dtype=bool)).realize()
    _export_and_compare(wrapper, inp, mask, rtol=1e-4, atol=1e-4)

  def test_detr_structure_export(self):
    """Export DETR structure model (ResNet18 + 6-layer transformer, 125 queries, 6 classes)."""
    model = self._build_detr(num_queries=125, num_classes=6)
    wrapper = self._make_wrapper(model)
    inp = Tensor(np.random.rand(1, 3, 128, 128).astype(np.float32)).realize()
    mask = Tensor(np.zeros((1, 128, 128), dtype=bool)).realize()
    _export_and_compare(wrapper, inp, mask, rtol=1e-4, atol=1e-4)

  def test_detr_emits_efficient_ops(self):
    """Verify the ONNX graph uses high-level ops (Conv, MatMul, Relu) not just element-wise."""
    from extra.export_onnx import export_onnx
    from collections import Counter
    model = self._build_detr(num_queries=15, num_classes=2)
    wrapper = self._make_wrapper(model)
    inp = Tensor(np.random.rand(1, 3, 128, 128).astype(np.float32)).realize()
    mask = Tensor(np.zeros((1, 128, 128), dtype=bool)).realize()
    onnx_model = export_onnx(wrapper, inp, mask)
    ops = Counter(n.op_type for n in onnx_model.graph.node)
    # ResNet18 backbone has 17 conv layers + input_proj
    self.assertGreaterEqual(ops.get("Conv", 0), 17)
    # Transformer has many MatMul ops (Q/K/V projections, attention, FFN)
    self.assertGreaterEqual(ops.get("MatMul", 0), 12)
    # ReLU activations in backbone
    self.assertGreaterEqual(ops.get("Relu", 0), 10)

  def test_detr_different_input_sizes(self):
    """Verify export works with different spatial dimensions."""
    model = self._build_detr(num_queries=15, num_classes=2)
    wrapper = self._make_wrapper(model)
    for h, w in [(64, 64), (128, 96), (96, 128)]:
      with self.subTest(h=h, w=w):
        inp = Tensor(np.random.rand(1, 3, h, w).astype(np.float32)).realize()
        mask = Tensor(np.zeros((1, h, w), dtype=bool)).realize()
        _export_and_compare(wrapper, inp, mask, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
  unittest.main()
