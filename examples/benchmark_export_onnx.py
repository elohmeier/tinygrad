#!/usr/bin/env python3
"""Benchmark tinygrad vs ONNX Runtime inference on CPU and CUDA.

Usage:
  PYTHONPATH=. python examples/benchmark_export_onnx.py
  PYTHONPATH=. python examples/benchmark_export_onnx.py --layers 4 --hidden 512 --batch 64
  PYTHONPATH=. python examples/benchmark_export_onnx.py --only-cpu
  PYTHONPATH=. python examples/benchmark_export_onnx.py --only-cuda
"""
import argparse, time, numpy as np

def build_model(layers: int, in_features: int, hidden: int, out_features: int, device: str):
  from tinygrad.tensor import Tensor
  weights, biases = [], []
  for i in range(layers):
    fan_in = in_features if i == 0 else hidden
    fan_out = out_features if i == layers - 1 else hidden
    weights.append(Tensor.kaiming_uniform(fan_out, fan_in, device=device).realize())
    biases.append(Tensor.zeros(fan_out, device=device).realize())

  def forward(x):
    for i, (w, b) in enumerate(zip(weights, biases)):
      x = x @ w.T + b
      if i < len(weights) - 1: x = x.relu()
    return x

  forward.weights = weights
  forward.biases = biases
  return forward

def bench_tinygrad(model, example_np: np.ndarray, device: str, warmup: int, runs: int) -> list[float]:
  from tinygrad.tensor import Tensor
  inp = Tensor(example_np, device=device).realize()

  # warmup
  for _ in range(warmup):
    model(inp).realize()

  # timed runs
  times = []
  for _ in range(runs):
    t0 = time.perf_counter()
    out = model(inp).realize()
    _ = out.numpy()  # force sync
    times.append(time.perf_counter() - t0)
  return times

def bench_onnxruntime(onnx_model, example_np: np.ndarray, provider: str, warmup: int, runs: int) -> list[float]:
  import onnxruntime as ort
  opts = ort.SessionOptions()
  opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  sess = ort.InferenceSession(onnx_model.SerializeToString(), opts, providers=[provider])
  feed = {sess.get_inputs()[0].name: example_np}

  for _ in range(warmup): sess.run(None, feed)

  times = []
  for _ in range(runs):
    t0 = time.perf_counter()
    sess.run(None, feed)
    times.append(time.perf_counter() - t0)
  return times

def print_stats(label: str, times: list[float]):
  arr = np.array(times) * 1e3
  print(f"  {label:30s}  median {np.median(arr):8.3f} ms  mean {np.mean(arr):8.3f} ms  "
        f"min {np.min(arr):8.3f} ms  std {np.std(arr):7.3f} ms  ({len(arr)} runs)")

def run_benchmarks(args, device: str, ort_provider: str):
  from tinygrad.tensor import Tensor
  from extra.export_onnx import export_onnx

  print(f"--- {device} ---")

  # build model on target device
  model = build_model(args.layers, args.in_features, args.hidden, args.out_features, device)
  example = Tensor.rand(args.batch, args.in_features, device=device).realize()
  example_np = example.numpy().astype(np.float32)

  # export to onnx
  onnx_model = export_onnx(model, example)

  # verify correctness
  import onnxruntime as ort
  sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=[ort_provider])
  ort_out = sess.run(None, {sess.get_inputs()[0].name: example_np})[0]
  expected = model(example).numpy()
  max_diff = np.max(np.abs(expected - ort_out))
  print(f"  correctness: max abs diff = {max_diff:.2e}")

  # benchmark
  tg_times = bench_tinygrad(model, example_np, device, args.warmup, args.runs)
  print_stats(f"tinygrad ({device})", tg_times)

  ort_times = bench_onnxruntime(onnx_model, example_np, ort_provider, args.warmup, args.runs)
  print_stats(f"ONNX Runtime ({ort_provider.split('Execution')[0]})", ort_times)

  speedup = np.median(tg_times) / np.median(ort_times)
  print(f"  {'ORT/tinygrad speedup':30s}  {speedup:.2f}x\n")

def main():
  parser = argparse.ArgumentParser(description="Benchmark tinygrad vs ONNX Runtime")
  parser.add_argument("--layers", type=int, default=4, help="number of linear layers")
  parser.add_argument("--hidden", type=int, default=256, help="hidden dimension")
  parser.add_argument("--in-features", type=int, default=256, help="input features")
  parser.add_argument("--out-features", type=int, default=128, help="output features")
  parser.add_argument("--batch", type=int, default=32, help="batch size")
  parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
  parser.add_argument("--runs", type=int, default=50, help="timed iterations")
  parser.add_argument("--only-cpu", action="store_true", help="skip CUDA benchmarks")
  parser.add_argument("--only-cuda", action="store_true", help="skip CPU benchmarks")
  parser.add_argument("--cnn", action="store_true", help="also benchmark a CNN model")
  args = parser.parse_args()

  import onnxruntime as ort

  print(f"Model: {args.layers}-layer MLP, in={args.in_features}, hidden={args.hidden}, out={args.out_features}, batch={args.batch}")
  print(f"Warmup: {args.warmup}, Runs: {args.runs}\n")

  if not args.only_cuda:
    run_benchmarks(args, "CPU", "CPUExecutionProvider")

  if not args.only_cpu:
    if "CUDAExecutionProvider" in ort.get_available_providers():
      run_benchmarks(args, "CUDA", "CUDAExecutionProvider")
    else:
      print("--- CUDA ---\n  ONNX Runtime CUDAExecutionProvider not available, skipping\n")

  # also benchmark CNN if requested
  if args.cnn:
    run_cnn_benchmark(args)

def build_cnn(device: str):
  from tinygrad.tensor import Tensor
  conv1 = Tensor.rand(32, 3, 3, 3, device=device).realize()
  conv2 = Tensor.rand(64, 32, 3, 3, device=device).realize()
  conv3 = Tensor.rand(128, 64, 3, 3, device=device).realize()
  fc = Tensor.rand(10, 128, device=device).realize()
  fc_b = Tensor.rand(10, device=device).realize()

  def forward(x):
    x = x.conv2d(conv1, padding=1).relu()
    x = x.conv2d(conv2, padding=1, stride=2).relu()
    x = x.conv2d(conv3, padding=1, stride=2).relu()
    x = x.mean(axis=(2, 3))
    return x @ fc.T + fc_b

  forward.weights = [conv1, conv2, conv3, fc, fc_b]
  return forward

def run_cnn_benchmark(args):
  from tinygrad.tensor import Tensor
  from extra.export_onnx import export_onnx
  import onnxruntime as ort

  device = "CPU"
  print(f"CNN Model: 3-conv + FC, input=(4, 3, 32, 32)\n")

  model = build_cnn(device)
  example = Tensor.rand(4, 3, 32, 32, device=device).realize()
  example_np = example.numpy().astype(np.float32)
  onnx_model = export_onnx(model, example)

  # verify correctness
  expected = model(example).numpy()
  sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
  ort_out = sess.run(None, {sess.get_inputs()[0].name: example_np})[0]
  print(f"  correctness: max abs diff = {np.max(np.abs(expected - ort_out)):.2e}")

  # count ONNX ops
  from collections import Counter
  ops = Counter(n.op_type for n in onnx_model.graph.node)
  print(f"  ONNX ops: {dict(ops)}")

  tg_times = bench_tinygrad(model, example_np, device, args.warmup, args.runs)
  print_stats(f"tinygrad ({device})", tg_times)

  ort_times = bench_onnxruntime(onnx_model, example_np, "CPUExecutionProvider", args.warmup, args.runs)
  print_stats(f"ONNX Runtime (CPU)", ort_times)

  speedup = np.median(tg_times) / np.median(ort_times)
  print(f"  {'ORT/tinygrad speedup':30s}  {speedup:.2f}x\n")

if __name__ == "__main__":
  main()
