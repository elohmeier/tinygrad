import math, numpy as np
from typing import Any
from tinygrad.tensor import Tensor
from tinygrad.dtype import DType, dtypes, PtrDType
from tinygrad.uop import Ops, GroupOp
from tinygrad.uop.ops import UOp, all_metadata

# lazy imports for onnx
def _import_onnx():
  import onnx
  from onnx import helper, TensorProto, numpy_helper
  return onnx, helper, TensorProto, numpy_helper

DTYPE_TO_ONNX: dict[DType, int] = {}
def _get_onnx_dtype(dt: DType) -> int:
  if not DTYPE_TO_ONNX:
    _, _, TensorProto, _ = _import_onnx()
    DTYPE_TO_ONNX.update({
      dtypes.float32: TensorProto.FLOAT, dtypes.float16: TensorProto.FLOAT16, dtypes.float64: TensorProto.DOUBLE,
      dtypes.int8: TensorProto.INT8, dtypes.int16: TensorProto.INT16, dtypes.int32: TensorProto.INT32, dtypes.int64: TensorProto.INT64,
      dtypes.uint8: TensorProto.UINT8, dtypes.uint16: TensorProto.UINT16, dtypes.uint32: TensorProto.UINT32, dtypes.uint64: TensorProto.UINT64,
      dtypes.bool: TensorProto.BOOL, dtypes.bfloat16: TensorProto.BFLOAT16,
    })
  base = dt.base if isinstance(dt, PtrDType) else dt
  if base not in DTYPE_TO_ONNX: raise RuntimeError(f"unsupported dtype for ONNX export: {base}")
  return DTYPE_TO_ONNX[base]

def _is_shape_uop(uop: UOp) -> bool:
  if uop.op in {Ops.VCONST, Ops.CONST} and hasattr(uop.dtype, 'scalar') and uop.dtype.scalar() == dtypes.weakint: return True
  if uop.op == Ops.VECTORIZE: return True
  return False

class OnnxBuilder:
  def __init__(self):
    self.nodes: list[Any] = []
    self.initializers: list[Any] = []
    self.graph_inputs: list[Any] = []
    self.graph_outputs: list[Any] = []
    self.uop_names: dict[int, str] = {}
    self._counter = 0

  def fresh_name(self, prefix: str = "t") -> str:
    name = f"{prefix}{self._counter}"
    self._counter += 1
    return name

  def name(self, uop: UOp) -> str:
    uid = id(uop)
    if uid not in self.uop_names: self.uop_names[uid] = self.fresh_name()
    return self.uop_names[uid]

  def set_name(self, uop: UOp, name: str): self.uop_names[id(uop)] = name

  def add_initializer(self, name: str, np_array: np.ndarray):
    _, _, _, numpy_helper = _import_onnx()
    self.initializers.append(numpy_helper.from_array(np_array, name=name))

  def add_node(self, op_type: str, inputs: list[str], outputs: list[str], **attrs):
    _, helper, _, _ = _import_onnx()
    self.nodes.append(helper.make_node(op_type, inputs, outputs, **attrs))

  def add_const_tensor(self, name: str, np_array: np.ndarray):
    self.add_initializer(name, np_array)

def _uop_dtype(uop: UOp) -> DType:
  return uop.dtype.base if isinstance(uop.dtype, PtrDType) else uop.dtype

# *** pattern matching for high-level ONNX ops ***

def _walk_reshape(u: UOp) -> UOp:
  """Walk through RESHAPE chain to find the data source."""
  while u.op == Ops.RESHAPE: u = u.src[0]
  return u

def _walk_through(u: UOp, *ops: Ops) -> UOp | None:
  """Walk through a chain of exactly the given ops (in order from outside in). Returns the innermost source or None."""
  for op in ops:
    if u.op != op: return None
    u = u.src[0]
  return u

def _match_matmul(uop: UOp, consumer_counts: dict[int, int]) -> tuple[UOp, UOp, bool, list[UOp]] | None:
  """Match the tinygrad matmul pattern rooted at a RESHAPE after REDUCE_AXIS.

  Pattern 1 (A @ B — PERMUTE on RHS):
    RESHAPE <- REDUCE_AXIS(ADD, last) <- MUL <- (EXPAND <- RESHAPE <- A, EXPAND <- PERMUTE <- RESHAPE <- B)

  Pattern 2 (A @ B^T — no PERMUTE, both sides EXPAND <- RESHAPE):
    RESHAPE <- REDUCE_AXIS(ADD, last) <- MUL <- (EXPAND <- RESHAPE <- A, EXPAND <- RESHAPE <- B)
    Here, both A and B have the contraction dim K as last, so the matmul is A @ B^T.

  Returns (lhs_data, rhs_data, transB, consumed_uops) where:
    - lhs_data: the UOp providing the left matrix
    - rhs_data: the UOp providing the right matrix
    - transB: True if rhs needs to be transposed (Pattern 2)
    - consumed_uops: all intermediate UOps that should be skipped in normal emit
  """
  # uop must be RESHAPE (the final shape collapse, e.g. (M,N,1) -> (M,N))
  if uop.op != Ops.RESHAPE: return None
  reduce = uop.src[0]
  if reduce.op != Ops.REDUCE_AXIS: return None
  reduce_op, axes = reduce.arg
  if reduce_op != Ops.ADD or len(axes) != 1: return None

  mul = reduce.src[0]
  if mul.op != Ops.MUL: return None

  lhs_expand, rhs_expand = mul.src[0], mul.src[1]
  if lhs_expand.op != Ops.EXPAND or rhs_expand.op != Ops.EXPAND: return None

  # both expands must produce the same shape (the broadcast matmul shape)
  if lhs_expand.shape != rhs_expand.shape: return None
  bc_shape = lhs_expand.shape
  reduce_axis = axes[0]
  ndim = len(bc_shape)

  # LHS: EXPAND <- RESHAPE <- data  (inserts a 1 in the N position)
  lhs_reshape = lhs_expand.src[0]
  if lhs_reshape.op != Ops.RESHAPE: return None

  # Try Pattern 1: RHS has PERMUTE (A @ B)
  rhs_next = rhs_expand.src[0]
  if rhs_next.op == Ops.PERMUTE:
    rhs_permute = rhs_next
    rhs_reshape = rhs_permute.src[0]
    if rhs_reshape.op != Ops.RESHAPE: return None

    lhs_pre = lhs_reshape.src[0]
    rhs_pre = rhs_reshape.src[0]

    # verify the permute swaps the last two dims
    perm = rhs_permute.marg
    expected_perm = list(range(ndim))
    expected_perm[-1], expected_perm[-2] = expected_perm[-2], expected_perm[-1]
    if list(perm) != expected_perm: return None

    # Only require single-use for the core matmul chain (MUL and REDUCE).
    # EXPAND/RESHAPE/PERMUTE may be shared between multiple matmuls (e.g., Q/K/V projections).
    if consumer_counts.get(id(reduce), 0) > 1: return None
    if consumer_counts.get(id(mul), 0) > 1: return None

    # Build consumed set: always include the core chain.
    # For shared expand/reshape nodes, don't consume them — they'll be emitted normally
    # and become dead code that ORT can optimize away.
    consumed = [uop, reduce, mul]
    for u in [lhs_expand, lhs_reshape, rhs_expand, rhs_permute, rhs_reshape]:
      if consumer_counts.get(id(u), 0) <= 1:
        consumed.append(u)
    return (lhs_pre, rhs_pre, False, consumed)

  # Try Pattern 2: RHS has no PERMUTE — both sides EXPAND <- RESHAPE (A @ B^T)
  # This happens in linear layers: input (M, 1, 1, K) and weight (1, 1, N, K) both
  # expanded to (M, 1, N, K), multiplied, then reduced on last axis K.
  if rhs_next.op == Ops.RESHAPE:
    rhs_reshape = rhs_next
    lhs_pre = lhs_reshape.src[0]
    rhs_pre = rhs_reshape.src[0]

    # For pattern 2, the reduce axis must be the last dim
    if reduce_axis != ndim - 1: return None

    # Only require single-use for the core matmul chain (MUL and REDUCE).
    if consumer_counts.get(id(reduce), 0) > 1: return None
    if consumer_counts.get(id(mul), 0) > 1: return None

    consumed = [uop, reduce, mul]
    for u in [lhs_expand, lhs_reshape, rhs_expand, rhs_reshape]:
      if consumer_counts.get(id(u), 0) <= 1:
        consumed.append(u)
    return (lhs_pre, rhs_pre, True, consumed)

  return None

def _match_gemm(uop: UOp, consumer_counts: dict[int, int]) -> tuple[UOp, UOp, UOp, bool, list[UOp]] | None:
  """Match matmul + bias add pattern: ADD(matmul_result, EXPAND <- RESHAPE <- bias).

  Returns (lhs, rhs, bias_data, transB, consumed_uops) or None.
  """
  if uop.op != Ops.ADD: return None

  # try both orderings: ADD(matmul, bias) or ADD(bias, matmul)
  for mm_idx, bias_idx in [(0, 1), (1, 0)]:
    mm_cand = uop.src[mm_idx]
    bias_expand = uop.src[bias_idx]

    # the matmul result should be a RESHAPE (the output of _match_matmul pattern)
    mm_result = _match_matmul(mm_cand, consumer_counts)
    if mm_result is None: continue

    lhs, rhs, transB, mm_consumed = mm_result

    # bias: EXPAND <- RESHAPE <- bias_data (broadcasting bias to match output)
    if bias_expand.op != Ops.EXPAND: continue
    bias_reshape = bias_expand.src[0]
    if bias_reshape.op != Ops.RESHAPE: continue
    bias_data = bias_reshape.src[0]

    # check intermediates are single-use
    for u in [bias_expand, bias_reshape]:
      if consumer_counts.get(id(u), 0) > 1: return None

    consumed = mm_consumed + [bias_expand, bias_reshape, uop]
    return (lhs, rhs, bias_data, transB, consumed)

  return None

def _find_conv_input_from_pad(uop: UOp) -> UOp | None:
  """For a padded conv, the PAD's source IS the input (either RESHAPE from BUFFER, or a computed tensor like relu output)."""
  if uop.op == Ops.PAD:
    meta = all_metadata.get(uop)
    if meta and meta[0].name == 'conv2d': return uop.src[0]
  meta = all_metadata.get(uop)
  if not meta or meta[0].name != 'conv2d': return None
  for s in uop.src:
    result = _find_conv_input_from_pad(s)
    if result is not None: return result
  return None

def _find_conv_input_no_pad(uop: UOp, weight_buf_size: int) -> UOp | None:
  """For unpadded conv, find the RESHAPE from non-weight BUFFER that starts the im2col chain.

  Also finds 4D inputs from previous layers (not just BUFFERs) by looking for
  the boundary where conv2d-tagged ops meet non-conv2d sources.
  """
  if uop.op == Ops.BUFFER: return None
  if uop.op == Ops.RESHAPE and len(uop.shape) == 4:
    src = uop.src[0]
    if src.op == Ops.BUFFER and int(src.arg) != weight_buf_size: return uop
  meta = all_metadata.get(uop)
  if not meta or meta[0].name != 'conv2d': return None
  for s in uop.src:
    result = _find_conv_input_no_pad(s, weight_buf_size)
    if result is not None: return result
  # Fallback: look for non-conv2d-tagged sources with exactly 4D shape
  for s in uop.src:
    s_meta = all_metadata.get(s)
    if (not s_meta or s_meta[0].name != 'conv2d') and s._shape is not None and len(s.shape) == 4:
      return s
  return None

def _find_conv_pad(uop: UOp) -> UOp | None:
  """Find a PAD UOp tagged conv2d in the backward slice."""
  if uop.op == Ops.PAD:
    meta = all_metadata.get(uop)
    if meta and meta[0].name == 'conv2d': return uop
  meta = all_metadata.get(uop)
  if not meta or meta[0].name != 'conv2d': return None
  for s in uop.src:
    result = _find_conv_pad(s)
    if result is not None: return result
  return None

def _find_stride(H_in: int, pad: int, kH: int, H_out: int) -> int:
  for s in range(1, max(kH, H_in) + 1):
    if (H_in + 2 * pad - kH) // s + 1 == H_out: return s
  return 1

def _match_conv2d(all_uops: dict[UOp, None]) -> list[tuple]:
  """Detect conv2d patterns using metadata tags.

  Returns list of (root_uop, input_uop, weight_uop, bias_uop_or_None, kernel_shape, strides, pads, groups, consumed_uops).
  """
  results = []
  # Find REDUCE_AXIS nodes tagged as conv2d
  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name != 'conv2d' or uop.op != Ops.REDUCE_AXIS: continue
    reduce_op, axes = uop.arg
    if reduce_op != Ops.ADD: continue

    mul = uop.src[0]
    if mul.op != Ops.MUL or len(mul.shape) != 8: continue

    # MUL shape: (N, groups_dim, C_out/g, H_out, W_out, C_in/g, kH, kW)
    ms = [int(x) for x in mul.shape]
    N, groups_dim, Co_g, Ho, Wo, Ci_g, kH, kW = ms

    # Find the final RESHAPE to 4D (output of conv)
    output_reshape = None
    for u in all_uops:
      if u.op == Ops.RESHAPE and u.src[0] is uop and len(u.shape) == 4:
        output_reshape = u
        break
    if output_reshape is None: continue

    # Find the bias ADD if present (ADD tagged conv2d whose src is the output_reshape or its expand)
    bias_add = None
    bias_data = None
    for u in all_uops:
      if u.op != Ops.ADD: continue
      u_meta = all_metadata.get(u)
      if not u_meta or u_meta[0].name != 'conv2d': continue
      # check if one of the ADD sources traces to our output_reshape
      for src_idx in range(2):
        other_idx = 1 - src_idx
        if u.src[src_idx] is output_reshape:
          # the other source should be the bias (EXPAND <- RESHAPE <- BUFFER)
          bias_expand = u.src[other_idx]
          if bias_expand.op == Ops.EXPAND and bias_expand.src[0].op == Ops.RESHAPE:
            bias_data = bias_expand.src[0].src[0]  # the BUFFER or data source
            bias_add = u
            break
      if bias_add is not None: break

    # Find padding from conv2d-tagged PAD
    pad_uop = _find_conv_pad(mul)
    if pad_uop is not None:
      # pads are ((0,0), (0,0), (pH, pH), (pW, pW)) for NCHW
      pad_h = int(pad_uop.marg[2][0])
      pad_w = int(pad_uop.marg[3][0])
    else:
      pad_h, pad_w = 0, 0

    # Find input: try padded path first, then unpadded
    weight_buf_size = groups_dim * Co_g * Ci_g * kH * kW
    input_data = _find_conv_input_from_pad(mul)  # returns PAD's source
    if input_data is None:
      input_data = _find_conv_input_no_pad(mul, weight_buf_size)
    if input_data is None: continue
    inp_shape = [int(x) for x in input_data.shape]
    H_in, W_in = inp_shape[2], inp_shape[3]

    # Find weight: trace through the other MUL operand to a BUFFER
    # Weight side: EXPAND <- RESHAPE <- RESHAPE <- BUFFER  (the RESHAPE gives it 4D: C_out, C_in/g, kH, kW)
    weight_data = None
    for src in mul.src:
      # walk through EXPAND/RESHAPE/PERMUTE to find a BUFFER
      u = src
      while u.op in {Ops.EXPAND, Ops.RESHAPE, Ops.PERMUTE}: u = u.src[0]
      if u.op == Ops.BUFFER:
        expected_weight_size = groups_dim * Co_g * Ci_g * kH * kW
        if int(u.arg) == expected_weight_size:
          weight_data = u
          break
    if weight_data is None: continue

    # Compute stride
    stride_h = _find_stride(H_in, pad_h, kH, Ho)
    stride_w = _find_stride(W_in, pad_w, kW, Wo)

    groups = groups_dim
    C_out = groups_dim * Co_g

    # Collect conv2d-tagged UOps as consumed:
    # Only include UOps in the reduce's backward slice that are NOT in the input_data's backward slice
    # (to avoid consuming UOps from previous layers that feed into this conv's input)
    reduce_topo = uop.toposort()
    input_topo = input_data.toposort() if input_data.op != Ops.BUFFER else {}
    preserved = {id(input_data), id(weight_data)}
    if bias_data is not None: preserved.add(id(bias_data))
    consumed_uops = []
    for u in reduce_topo:
      if id(u) in preserved: continue
      if u in input_topo or u is input_data: continue  # belongs to previous layer
      u_meta = all_metadata.get(u)
      if u_meta and u_meta[0].name == 'conv2d':
        consumed_uops.append(u)
    # also consume the output_reshape and bias_add
    consumed_uops.append(output_reshape)
    if bias_add is not None:
      consumed_uops.append(bias_add)
      for u in bias_add.src:
        if u is not output_reshape:
          u_meta = all_metadata.get(u)
          if u_meta and u_meta[0].name == 'conv2d':
            consumed_uops.append(u)
            if u.op == Ops.EXPAND and u.src[0].op == Ops.RESHAPE:
              u2_meta = all_metadata.get(u.src[0])
              if u2_meta and u2_meta[0].name == 'conv2d' and id(u.src[0]) not in preserved:
                consumed_uops.append(u.src[0])
    root = bias_add if bias_add is not None else output_reshape

    results.append((root, input_data, weight_data, bias_data, [kH, kW], [stride_h, stride_w],
                     [pad_h, pad_w, pad_h, pad_w], groups, consumed_uops))

  return results

def _walk_tagged(uop: UOp, meta_name: str, visited: set[int] | None = None) -> set[int]:
  """Walk backward from uop collecting only same-tagged UOps. Stops at non-tagged boundaries."""
  if visited is None: visited = set()
  if id(uop) in visited: return visited
  visited.add(id(uop))
  for s in uop.src:
    s_meta = all_metadata.get(s)
    if s_meta and s_meta[0].name == meta_name:
      _walk_tagged(s, meta_name, visited)
  return visited

def _match_metadata_pattern(all_uops: dict[UOp, None], meta_name: str, onnx_op: str,
                             output_uop: UOp, attrs: dict | None = None) -> tuple[UOp, UOp, list[UOp], str, dict] | None:
  """Generic matcher for metadata-tagged patterns (activations, etc.).

  Walks backward from output_uop through same-tagged UOps only (stops at non-tagged boundaries).
  Identifies the input (first non-tagged source with matching shape) and consumes only
  same-tagged UOps that have no external consumers.
  """
  # walk backward only through same-tagged UOps
  tagged_ids = _walk_tagged(output_uop, meta_name)
  tagged_uops = [u for u in all_uops if id(u) in tagged_ids]

  # find the input: first non-tagged source with matching shape
  input_data = None
  for u in tagged_uops:
    for s in u.src:
      s_meta = all_metadata.get(s)
      if (not s_meta or s_meta[0].name != meta_name) and s._shape is not None and s.shape == output_uop.shape:
        input_data = s
        break
    if input_data is not None: break

  if input_data is None: return None

  # Don't consume intermediate UOps for activations — they may be shared between
  # multiple activation instances (e.g., two relus sharing the zero constant broadcast).
  # The intermediate ops will emit as dead code that ORT optimizes away.
  # Only the root itself is consumed (via the pattern dict).
  return (output_uop, input_data, [], onnx_op, attrs or {})

def _match_activations(all_uops: dict[UOp, None]) -> list[tuple]:
  """Detect activation patterns using metadata tags.

  Returns list of (root_uop, input_uop, consumed_uops, onnx_op, attrs).
  """
  results = []
  activation_map = {"relu": "Relu", "sigmoid": "Sigmoid", "tanh": "Tanh", "gelu": "Gelu"}

  # build consumer map: for each tagged uop, track if any same-tagged uop consumes it
  has_tagged_consumer: set[int] = set()
  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name not in activation_map: continue
    for s in uop.src:
      s_meta = all_metadata.get(s)
      if s_meta and s_meta[0].name == meta[0].name:
        has_tagged_consumer.add(id(s))

  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name not in activation_map: continue
    if id(uop) in has_tagged_consumer: continue  # not a root

    result = _match_metadata_pattern(all_uops, meta[0].name, activation_map[meta[0].name], uop)
    if result is not None: results.append(result)

  return results

def _match_softmax(all_uops: dict[UOp, None]) -> list[tuple]:
  """Detect softmax patterns. Need to extract the axis from the REDUCE_AXIS."""
  results = []
  # find roots (no same-tagged consumer)
  has_tagged_consumer: set[int] = set()
  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name != 'softmax': continue
    for s in uop.src:
      s_meta = all_metadata.get(s)
      if s_meta and s_meta[0].name == 'softmax':
        has_tagged_consumer.add(id(s))

  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name != 'softmax': continue
    if id(uop) in has_tagged_consumer: continue

    # find the axis from the first REDUCE_AXIS
    axis = -1
    for u in uop.toposort():
      u_meta = all_metadata.get(u)
      if u_meta and u_meta[0].name == 'softmax' and u.op == Ops.REDUCE_AXIS:
        axis = u.arg[1][0]
        break

    result = _match_metadata_pattern(all_uops, 'softmax', 'Softmax', uop, {"axis": axis})
    if result is not None: results.append(result)

  return results

def _match_pooling(all_uops: dict[UOp, None]) -> list[tuple]:
  """Detect max_pool2d and avg_pool2d patterns.

  Pool pattern ends with: PERMUTE -> REDUCE_AXIS(MAX or ADD, last 2 axes) -> RESHAPE(4D)
  The PERMUTE shape is (N, C, H_out, W_out, kH, kW).
  """
  results = []
  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name not in ('max_pool2d', 'avg_pool2d'): continue
    if uop.op != Ops.REDUCE_AXIS: continue

    pool_name = meta[0].name
    reduce_op, axes = uop.arg

    # verify this is a pool reduce: reduces last 2 dims of a 6D tensor
    if len(uop.src[0].shape) != 6 or axes != (4, 5): continue
    perm_shape = uop.src[0].shape  # after PERMUTE: (N, C, H_out, W_out, kH, kW)
    N, C, Ho, Wo, kH, kW = [int(x) for x in perm_shape]

    # find the output RESHAPE (4D)
    output_reshape = None
    for u in all_uops:
      if u.op == Ops.RESHAPE and u.src[0] is uop and len(u.shape) == 4:
        output_reshape = u
        break

    # for avg_pool, the root is the MUL (multiply by 1/k²) after RESHAPE
    root = output_reshape
    if pool_name == 'avg_pool2d' and output_reshape is not None:
      for u in all_uops:
        u_meta = all_metadata.get(u)
        if u_meta and u_meta[0].name == 'avg_pool2d' and u.op == Ops.MUL:
          # check if one src is the output_reshape
          if output_reshape in u.src:
            root = u
            break

    if root is None: continue

    # find input: look for PAD (padded pool) or the first non-tagged source
    pad_uop = _find_conv_pad(uop)  # reuse conv helper
    if pad_uop is not None:
      input_data = pad_uop.src[0]
      pad_h, pad_w = int(pad_uop.marg[2][0]), int(pad_uop.marg[3][0])
    else:
      # find first RESHAPE from non-tagged source
      input_data = None
      for u in uop.toposort():
        u_meta = all_metadata.get(u)
        if u_meta and u_meta[0].name == pool_name and u.op == Ops.RESHAPE and len(u.shape) == 4:
          src = u.src[0]
          src_meta = all_metadata.get(src)
          if not src_meta or src_meta[0].name != pool_name:
            input_data = u.src[0] if u.src[0].op != Ops.BUFFER else u
            break
      if input_data is None: continue
      pad_h, pad_w = 0, 0

    if len(input_data.shape) < 4: continue
    H_in, W_in = int(input_data.shape[2]), int(input_data.shape[3])
    stride_h = _find_stride(H_in, pad_h, kH, Ho)
    stride_w = _find_stride(W_in, pad_w, kW, Wo)

    onnx_op = "MaxPool" if pool_name == 'max_pool2d' else "AveragePool"
    attrs = {"kernel_shape": [kH, kW], "strides": [stride_h, stride_w], "pads": [pad_h, pad_w, pad_h, pad_w]}

    # collect consumed UOps
    reduce_topo = uop.toposort()
    input_topo = input_data.toposort() if input_data.op != Ops.BUFFER else {}
    consumed_uops = []
    for u in all_uops:
      u_meta = all_metadata.get(u)
      if not u_meta or u_meta[0].name != pool_name: continue
      if u in input_topo or u is input_data: continue
      if u in reduce_topo or u is uop or u is output_reshape or u is root:
        consumed_uops.append(u)

    results.append((root, input_data, consumed_uops, onnx_op, attrs))

  return results

def _match_interpolate(all_uops: dict[UOp, None]) -> list[tuple]:
  """Detect interpolate patterns (nearest/linear upsampling).

  Returns list of (root_uop, input_uop, consumed_uops, onnx_op, attrs).
  """
  results = []
  # find roots: tagged 'interpolate' with no same-tagged consumer
  has_tagged_consumer: set[int] = set()
  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name != 'interpolate': continue
    for s in uop.src:
      s_meta = all_metadata.get(s)
      if s_meta and s_meta[0].name == 'interpolate':
        has_tagged_consumer.add(id(s))

  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name != 'interpolate': continue
    if id(uop) in has_tagged_consumer: continue

    output_shape = uop.shape
    if len(output_shape) != 4: continue

    # find input: first non-tagged source with 4D shape
    input_data = None
    for u in uop.toposort():
      u_meta = all_metadata.get(u)
      if not u_meta or u_meta[0].name != 'interpolate': continue
      for s in u.src:
        s_meta = all_metadata.get(s)
        if (not s_meta or s_meta[0].name != 'interpolate') and s._shape is not None and len(s.shape) == 4:
          input_data = s
          break
      if input_data is not None: break
    if input_data is None: continue

    # determine mode: linear has TRUNC ops in the tagged set, nearest doesn't
    has_trunc = any(u.op == Ops.TRUNC and all_metadata.get(u) and all_metadata.get(u)[0].name == 'interpolate'
                    for u in uop.toposort())
    mode = "linear" if has_trunc else "nearest"

    # output spatial dims
    out_h, out_w = int(output_shape[2]), int(output_shape[3])

    attrs = {"mode": mode, "output_size": [int(output_shape[0]), int(output_shape[1]), out_h, out_w]}
    if mode == "linear":
      attrs["coordinate_transformation_mode"] = "half_pixel"
    else:
      attrs["coordinate_transformation_mode"] = "asymmetric"
      attrs["nearest_mode"] = "floor"

    # collect consumed UOps (scoped to backward slice, excluding input's slice)
    input_topo = input_data.toposort() if input_data.op != Ops.BUFFER else {}
    consumed_uops = []
    for u in uop.toposort():
      u_meta = all_metadata.get(u)
      if u_meta and u_meta[0].name == 'interpolate' and u not in input_topo and u is not input_data:
        consumed_uops.append(u)
    # include root
    if uop not in consumed_uops: consumed_uops.append(uop)

    results.append((uop, input_data, consumed_uops, "Resize", attrs))

  return results

def _match_conv_transpose2d(all_uops: dict[UOp, None]) -> list[tuple]:
  """Detect conv_transpose2d patterns using metadata tags.

  Same MUL shape as conv2d: (N, groups, C_out, H_out, W_out, C_in, kH, kW)
  Weight has FLIP applied. Uses same parameter extraction as conv2d.
  """
  results = []
  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name != 'conv_transpose2d' or uop.op != Ops.REDUCE_AXIS: continue
    reduce_op, axes = uop.arg
    if reduce_op != Ops.ADD: continue

    mul = uop.src[0]
    if mul.op != Ops.MUL or len(mul.shape) != 8: continue

    ms = [int(x) for x in mul.shape]
    N, groups_dim, Co_g, Ho, Wo, Ci_g, kH, kW = ms

    # find output RESHAPE (4D)
    output_reshape = None
    for u in all_uops:
      if u.op == Ops.RESHAPE and u.src[0] is uop and len(u.shape) == 4:
        output_reshape = u
        break
    if output_reshape is None: continue

    # find weight BUFFER (through FLIP)
    weight_buf_size = groups_dim * Co_g * Ci_g * kH * kW
    weight_data = None
    for src in mul.src:
      u = src
      while u.op in {Ops.EXPAND, Ops.RESHAPE, Ops.PERMUTE, Ops.FLIP}: u = u.src[0]
      if u.op == Ops.BUFFER and int(u.arg) == weight_buf_size:
        weight_data = u
        break
    if weight_data is None: continue

    # find input
    input_data = _find_conv_input_from_pad(mul)
    if input_data is None:
      input_data = _find_conv_input_no_pad(mul, weight_buf_size)
    if input_data is None: continue

    inp_shape = [int(x) for x in input_data.shape]
    H_in, W_in = inp_shape[2], inp_shape[3]

    # For conv_transpose: output = (input - 1) * stride - 2*pad + kernel
    # Find PAD to determine padding
    pad_uop = _find_conv_pad(mul)
    if pad_uop is not None:
      # The PAD in conv_transpose is for stride insertion + boundary, more complex
      # Use output formula: Ho = (H_in - 1) * stride - 2*pad + kH
      # Try strides to find match
      pad_h = int(pad_uop.marg[2][0]) if len(pad_uop.marg) > 2 else 0
      pad_w = int(pad_uop.marg[3][0]) if len(pad_uop.marg) > 3 else 0
    else:
      pad_h, pad_w = 0, 0

    # compute stride: Ho = (H_in - 1) * stride - 2*output_pad + kH (with output_padding=0, pad from ONNX perspective)
    # Try strides 1..kH
    stride_h, stride_w = 1, 1
    for s in range(1, max(kH, Ho) + 1):
      # ONNX ConvTranspose: output_size = (input_size - 1) * stride + kernel_size - 2 * padding
      for p in range(kH):
        if (H_in - 1) * s + kH - 2 * p == Ho:
          stride_h = s
          pad_h = p
          break
    for s in range(1, max(kW, Wo) + 1):
      for p in range(kW):
        if (W_in - 1) * s + kW - 2 * p == Wo:
          stride_w = s
          pad_w = p
          break

    groups = groups_dim
    root = output_reshape

    # collect consumed UOps
    reduce_topo = uop.toposort()
    input_topo = input_data.toposort() if input_data.op != Ops.BUFFER else {}
    preserved = {id(input_data), id(weight_data)}
    consumed_uops = []
    for u in reduce_topo:
      if id(u) in preserved: continue
      if u in input_topo or u is input_data: continue
      u_meta = all_metadata.get(u)
      if u_meta and u_meta[0].name == 'conv_transpose2d':
        consumed_uops.append(u)
    consumed_uops.append(output_reshape)

    results.append((root, input_data, weight_data, None, [kH, kW], [stride_h, stride_w],
                     [pad_h, pad_w, pad_h, pad_w], groups, consumed_uops))

  return results

def _match_batchnorm(all_uops: dict[UOp, None]) -> list[tuple]:
  """Detect BatchNormalization patterns using metadata tags.

  tinygrad batchnorm decomposes to: output = weight * (input - running_mean) * rsqrt(running_var + eps) + bias
  The tagged UOps are: batchnorm (main chain) + rsqrt (sqrt/reciprocal) + surrounding reshape/expand/add.

  Returns list of (root_uop, input_uop, scale_buf, bias_buf, mean_buf, var_buf, epsilon, consumed_uops).
  """
  results = []

  # Find batchnorm roots (no same-tagged consumer)
  has_tagged_consumer: set[int] = set()
  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name != 'batchnorm': continue
    for s in uop.src:
      s_meta = all_metadata.get(s)
      if s_meta and s_meta[0].name == 'batchnorm':
        has_tagged_consumer.add(id(s))

  for root in all_uops:
    meta = all_metadata.get(root)
    if not meta or meta[0].name != 'batchnorm': continue
    if id(root) in has_tagged_consumer: continue

    # Collect all batchnorm-tagged UOps in backward slice
    bn_ids = _walk_tagged(root, 'batchnorm')
    bn_uops = [u for u in all_uops if id(u) in bn_ids]

    # Find the input: the non-batchnorm source with matching shape (typically from conv2d)
    input_data = None
    for u in bn_uops:
      for s in u.src:
        s_meta = all_metadata.get(s)
        if (not s_meta or s_meta[0].name != 'batchnorm') and s._shape is not None and s.shape == root.shape:
          if s.op != Ops.VCONST and s.op != Ops.CONST:
            input_data = s
            break
      if input_data is not None: break
    if input_data is None: continue

    # Find BUFFER sources (weight, bias, running_mean) — all are 1D buffers accessed via RESHAPE
    buffers: list[UOp] = []
    for u in bn_uops:
      if u.op == Ops.RESHAPE:
        for s in u.src:
          if s.op == Ops.BUFFER and s not in buffers:
            buffers.append(s)

    # Find the rsqrt chain to get running_var and epsilon
    var_buf = None
    epsilon = 1e-5
    for u in bn_uops:
      for s in u.src:
        if s.op == Ops.RECIPROCAL:
          # Walk into rsqrt-tagged subgraph
          rsqrt_ids = _walk_tagged(s, 'rsqrt')
          rsqrt_uops = [u2 for u2 in all_uops if id(u2) in rsqrt_ids]
          # Find the ADD(var, eps) that feeds rsqrt — it's a non-rsqrt source
          for u2 in rsqrt_uops:
            for s2 in u2.src:
              s2_meta = all_metadata.get(s2)
              if s2_meta and s2_meta[0].name == 'rsqrt': continue
              # This should be the ADD node for var+eps
              if s2.op == Ops.ADD and s2._shape is not None:
                # One source is var (via EXPAND<-RESHAPE<-BUFFER), other is eps (via EXPAND<-RESHAPE<-CONST)
                for s3 in s2.src:
                  if s3.op == Ops.EXPAND:
                    inner = s3.src[0] if s3.src else None
                    if inner is not None and inner.op == Ops.RESHAPE:
                      inner2 = inner.src[0] if inner.src else None
                      if inner2 is not None:
                        if inner2.op == Ops.BUFFER:
                          var_buf = inner2
                        elif inner2.op == Ops.CONST:
                          epsilon = float(inner2.arg)

    if var_buf is None: continue

    # We should have 3 direct BUFFER sources (running_mean, weight, bias) + var_buf
    # Determine which buffer is which based on the BN formula structure:
    # The BN pattern is: bias + weight * (input - mean) * rsqrt(var + eps)
    # root = ADD(MUL_chain, bias_expanded)
    # We identify them by tracing the root ADD's structure.

    # Identify bias: it's added last (root is ADD, one source is the bias expand chain)
    bias_buf = None
    scale_buf = None
    mean_buf = None

    # The BN root is an ADD. One source is the scaled (input-mean)*rsqrt*weight, the other is bias.
    # The bias side: EXPAND <- RESHAPE <- BUFFER
    if root.op == Ops.ADD:
      for src in root.src:
        if src.op == Ops.EXPAND:
          inner = src.src[0] if src.src else None
          if inner is not None and inner.op == Ops.RESHAPE:
            inner2 = inner.src[0] if inner.src else None
            if inner2 is not None and inner2.op == Ops.BUFFER:
              bias_buf = inner2

    # Now find mean and weight among remaining buffers
    remaining = [b for b in buffers if b is not bias_buf and b is not var_buf]

    # The mean is subtracted from input: look for ADD(input, MUL(-1, mean_expanded))
    # or SUB(input, mean_expanded). In the tagged chain, find which buffer is used
    # in a subtraction pattern.
    # Strategy: the buffer involved in an ADD with a MUL by -1 (via EXPAND<-RESHAPE<-CONST(-1)) is the mean.
    # The remaining one is the weight (scale).
    if len(remaining) == 2:
      # Check which is used in a multiply with the input shape vs which is the scale
      # Mean is in a MUL with EXPAND<-RESHAPE<-CONST (the -1 factor for subtraction)
      # Scale (weight) is in a MUL with the rsqrt result
      # Simpler approach: in BN, the formula is weight * (input - mean) / sqrt(var+eps) + bias
      # Tinygrad implements x - mean as x + (-1 * mean), so:
      # Look for MUL(EXPAND(RESHAPE(BUFFER)), EXPAND(RESHAPE(CONST))) where CONST is -1 or negative
      for u in bn_uops:
        if u.op == Ops.MUL:
          for s_idx in range(2):
            s = u.src[s_idx]
            other = u.src[1 - s_idx]
            if s.op == Ops.EXPAND and s.src[0].op == Ops.RESHAPE:
              buf_candidate = s.src[0].src[0] if s.src[0].src else None
              if buf_candidate in remaining:
                # Check if the other is a constant-like broadcast (the -1 factor for mean)
                if other.op == Ops.EXPAND and other.src[0].op == Ops.RESHAPE:
                  const_candidate = other.src[0].src[0] if other.src[0].src else None
                  if const_candidate is not None and const_candidate.op == Ops.CONST:
                    mean_buf = buf_candidate
                    break
          if mean_buf is not None: break

      if mean_buf is not None:
        scale_buf = [b for b in remaining if b is not mean_buf][0] if len(remaining) == 2 else None
      else:
        # Fallback: assign by convention (first = mean, second = scale)
        mean_buf = remaining[0]
        scale_buf = remaining[1] if len(remaining) > 1 else None
    elif len(remaining) == 1:
      # Only one remaining — likely mean (scale might be folded)
      mean_buf = remaining[0]

    if scale_buf is None or mean_buf is None or bias_buf is None: continue

    # Collect consumed UOps: all batchnorm-tagged + rsqrt-tagged + surrounding add/expand/reshape
    # IMPORTANT: exclude the input_data node — it may be shared between multiple BN patterns.
    consumed_uops = [u for u in bn_uops if u is not input_data]
    # Also consume rsqrt chain and the ADD(var, eps) and its sources
    for u in bn_uops:
      for s in u.src:
        if s.op == Ops.RECIPROCAL:
          rsqrt_ids = _walk_tagged(s, 'rsqrt')
          consumed_uops.extend(u2 for u2 in all_uops if id(u2) in rsqrt_ids)
    # Also consume the 'add' and 'expand' and 'reshape' tagged intermediates feeding into the rsqrt
    for u in list(consumed_uops):
      for s in u.src:
        s_meta = all_metadata.get(s)
        if s_meta and s_meta[0].name in ('add', 'expand', 'reshape'):
          add_ids = _walk_tagged(s, s_meta[0].name)
          consumed_uops.extend(u2 for u2 in all_uops if id(u2) in add_ids)

    results.append((root, input_data, scale_buf, bias_buf, mean_buf, var_buf, epsilon, consumed_uops))

  return results

def _match_global_avg_pool(all_uops: dict[UOp, None]) -> list[tuple]:
  """Detect mean (global average pooling) patterns using metadata tags.

  Pattern: REDUCE_AXIS(ADD, spatial_axes) → RESHAPE → MUL(1/N) = GlobalAveragePool
  Returns list of (root_uop, input_uop, consumed_uops).
  """
  results = []

  has_tagged_consumer: set[int] = set()
  for uop in all_uops:
    meta = all_metadata.get(uop)
    if not meta or meta[0].name != 'mean': continue
    for s in uop.src:
      s_meta = all_metadata.get(s)
      if s_meta and s_meta[0].name == 'mean':
        has_tagged_consumer.add(id(s))

  for root in all_uops:
    meta = all_metadata.get(root)
    if not meta or meta[0].name != 'mean': continue
    if id(root) in has_tagged_consumer: continue

    # Collect all mean-tagged UOps
    mean_ids = _walk_tagged(root, 'mean')
    mean_uops = [u for u in all_uops if id(u) in mean_ids]

    # Find input: non-mean-tagged source (typically from relu)
    input_data = None
    for u in mean_uops:
      if u.op == Ops.REDUCE_AXIS:
        for s in u.src:
          s_meta = all_metadata.get(s)
          if not s_meta or s_meta[0].name != 'mean':
            input_data = s
            break
      if input_data is not None: break
    if input_data is None: continue

    # Verify this is a spatial reduction (axes 2,3 for NCHW)
    reduce_uop = None
    for u in mean_uops:
      if u.op == Ops.REDUCE_AXIS:
        reduce_uop = u
        break
    if reduce_uop is None: continue
    _, axes = reduce_uop.arg
    if set(axes) != {2, 3} and len(input_data.shape) != 4: continue

    results.append((root, input_data, mean_uops))

  return results

def _count_consumers(all_uops: dict[UOp, None]) -> dict[int, int]:
  """Count how many UOps reference each UOp as a source."""
  counts: dict[int, int] = {}
  for uop in all_uops:
    for src in uop.src:
      uid = id(src)
      counts[uid] = counts.get(uid, 0) + 1
  return counts

def _detect_patterns(all_uops: dict[UOp, None]) -> tuple[dict[int, tuple], set[int]]:
  """Scan the UOp graph for high-level patterns (conv2d, matmul, gemm).

  Returns: dict mapping id(uop) to pattern info for the "root" UOp of each pattern,
           plus set of consumed UOp ids.
  """
  consumer_counts = _count_consumers(all_uops)
  patterns: dict[int, tuple] = {}
  consumed: set[int] = set()
  # Core computation UOps (REDUCE_AXIS, MUL from matmul chains) that must never
  # be un-consumed by post-processing.  Without this, un-consuming a shared
  # RESHAPE cascades and re-emits the full Expand→Mul→ReduceSum chain alongside
  # the matched MatMul/Gemm, causing massive memory blowup in ORT.
  _pattern_core: set[int] = set()

  # first: detect conv2d patterns (highest priority - subsumes matmul-like reduce patterns)
  for root, inp, weight, bias, kernel_shape, strides, pads, groups, consumed_uops in _match_conv2d(all_uops):
    if any(id(u) in consumed for u in consumed_uops): continue
    patterns[id(root)] = ("Conv", inp, weight, bias, kernel_shape, strides, pads, groups)
    for u in consumed_uops: consumed.add(id(u))

  # conv_transpose2d patterns
  for root, inp, weight, bias, kernel_shape, strides, pads, groups, consumed_uops in _match_conv_transpose2d(all_uops):
    if any(id(u) in consumed for u in consumed_uops): continue
    patterns[id(root)] = ("ConvTranspose", inp, weight, bias, kernel_shape, strides, pads, groups)
    for u in consumed_uops: consumed.add(id(u))

  uop_list = list(all_uops)

  # then Gemm (matmul+bias)
  for uop in uop_list:
    if id(uop) in consumed: continue
    result = _match_gemm(uop, consumer_counts)
    if result is not None:
      lhs, rhs, bias, transB, consumed_uops = result
      if any(id(u) in consumed for u in consumed_uops): continue
      patterns[id(uop)] = ("Gemm", lhs, rhs, bias, transB)
      for u in consumed_uops:
        consumed.add(id(u))
        if u.op in (Ops.REDUCE_AXIS, Ops.MUL): _pattern_core.add(id(u))

  # then plain MatMul
  for uop in uop_list:
    if id(uop) in consumed: continue
    result = _match_matmul(uop, consumer_counts)
    if result is not None:
      lhs, rhs, transB, consumed_uops = result
      # Filter already-consumed intermediates instead of rejecting the match.
      # Shared RESHAPE/EXPAND nodes (e.g. Q/K/V projections feeding both Gemm
      # and attention MatMul) should not block the MatMul from being emitted.
      filtered = [u for u in consumed_uops if id(u) not in consumed]
      patterns[id(uop)] = ("MatMul", lhs, rhs, transB)
      for u in filtered:
        consumed.add(id(u))
        if u.op in (Ops.REDUCE_AXIS, Ops.MUL): _pattern_core.add(id(u))

  # pooling patterns
  for root, inp, consumed_uops, onnx_op, attrs in _match_pooling(all_uops):
    if any(id(u) in consumed for u in consumed_uops): continue
    patterns[id(root)] = ("Pool", inp, onnx_op, attrs)
    for u in consumed_uops: consumed.add(id(u))

  # batchnorm patterns
  for root, inp, scale, bias, mean, var, eps, consumed_uops in _match_batchnorm(all_uops):
    if id(root) in consumed: continue
    filtered = [u for u in consumed_uops if id(u) not in consumed]
    patterns[id(root)] = ("BatchNorm", inp, scale, bias, mean, var, eps)
    for u in filtered: consumed.add(id(u))

  # global average pooling (mean over spatial dims)
  for root, inp, consumed_uops in _match_global_avg_pool(all_uops):
    if id(root) in consumed: continue
    filtered = [u for u in consumed_uops if id(u) not in consumed]
    patterns[id(root)] = ("GlobalAvgPool", inp)
    for u in filtered: consumed.add(id(u))

  # activation patterns (relu, sigmoid, tanh, gelu) — filter already-consumed UOps instead of skipping
  for root, inp, consumed_uops, onnx_op, attrs in _match_activations(all_uops):
    if id(root) in consumed: continue
    filtered = [u for u in consumed_uops if id(u) not in consumed]
    patterns[id(root)] = ("Activation", inp, onnx_op, attrs)
    for u in filtered: consumed.add(id(u))

  # softmax
  for root, inp, consumed_uops, onnx_op, attrs in _match_softmax(all_uops):
    if id(root) in consumed: continue
    filtered = [u for u in consumed_uops if id(u) not in consumed]
    patterns[id(root)] = ("Activation", inp, onnx_op, attrs)
    for u in filtered: consumed.add(id(u))

  # interpolate (Resize) patterns
  for root, inp, consumed_uops, onnx_op, attrs in _match_interpolate(all_uops):
    if any(id(u) in consumed for u in consumed_uops): continue
    patterns[id(root)] = ("Resize", inp, attrs)
    for u in consumed_uops: consumed.add(id(u))

  # Post-process: un-consume any node whose output is still needed by a live (non-consumed) node.
  # Build reverse map: for each uop, which uops consume it?
  consumers_of: dict[int, list[int]] = {}
  for uop in all_uops:
    for src in uop.src:
      consumers_of.setdefault(id(src), []).append(id(uop))

  # Iteratively un-consume nodes that have live consumers
  changed = True
  while changed:
    changed = False
    to_remove = []
    for uid in consumed:
      if uid in patterns: continue  # never un-consume pattern roots
      if uid in _pattern_core: continue  # never un-consume matmul core (REDUCE_AXIS, MUL)
      for consumer_id in consumers_of.get(uid, []):
        if consumer_id not in consumed:
          to_remove.append(uid)
          break
    for uid in to_remove:
      consumed.discard(uid)
      changed = True

  return patterns, consumed

# *** main emit logic ***

def _emit_uop(builder: OnnxBuilder, uop: UOp, patterns: dict[int, tuple], consumed: set[int]):
  uid = id(uop)

  # skip UOps consumed by patterns
  if uid in consumed and uid not in patterns: return

  # emit pattern if this is a pattern root
  if uid in patterns:
    pat = patterns[uid]
    out = builder.name(uop)
    if pat[0] == "Conv":
      _, inp, weight, bias, kernel_shape, strides, pads, groups = pat
      inputs = [builder.name(inp), builder.name(weight)]
      if bias is not None: inputs.append(builder.name(bias))
      builder.add_node("Conv", inputs, [out], kernel_shape=kernel_shape, strides=strides, pads=pads, group=groups)
      return
    if pat[0] == "MatMul":
      _, lhs, rhs, transB = pat
      if transB:
        # A @ B^T: emit Transpose on rhs, then MatMul
        trans_name = builder.fresh_name("trans_")
        rhs_ndim = len(rhs.shape) if hasattr(rhs, 'shape') and rhs.shape else 2
        perm = list(range(rhs_ndim))
        if rhs_ndim >= 2: perm[-1], perm[-2] = perm[-2], perm[-1]
        builder.add_node("Transpose", [builder.name(rhs)], [trans_name], perm=perm)
        builder.add_node("MatMul", [builder.name(lhs), trans_name], [out])
      else:
        builder.add_node("MatMul", [builder.name(lhs), builder.name(rhs)], [out])
    elif pat[0] == "Gemm":
      _, lhs, rhs, bias, transB = pat
      # ONNX Gemm: Y = alpha * A @ B + beta * C. Default alpha=1, beta=1.
      # But Gemm only supports 2D inputs. For batched, use MatMul + Add.
      lhs_shape = lhs.shape
      if len(lhs_shape) > 2:
        # batched: emit MatMul + Add (with optional transpose)
        mm_name = builder.fresh_name("mm_")
        if transB:
          trans_name = builder.fresh_name("trans_")
          rhs_ndim = len(rhs.shape) if hasattr(rhs, 'shape') and rhs.shape else 2
          perm = list(range(rhs_ndim))
          if rhs_ndim >= 2: perm[-1], perm[-2] = perm[-2], perm[-1]
          builder.add_node("Transpose", [builder.name(rhs)], [trans_name], perm=perm)
          builder.add_node("MatMul", [builder.name(lhs), trans_name], [mm_name])
        else:
          builder.add_node("MatMul", [builder.name(lhs), builder.name(rhs)], [mm_name])
        builder.add_node("Add", [mm_name, builder.name(bias)], [out])
      else:
        if transB:
          builder.add_node("Gemm", [builder.name(lhs), builder.name(rhs), builder.name(bias)], [out], transB=1)
        elif rhs.op == Ops.PERMUTE and rhs.marg == (1, 0):
          builder.add_node("Gemm", [builder.name(lhs), builder.name(rhs.src[0]), builder.name(bias)], [out], transB=1)
        else:
          builder.add_node("Gemm", [builder.name(lhs), builder.name(rhs), builder.name(bias)], [out])
    elif pat[0] == "ConvTranspose":
      _, inp, weight, bias, kernel_shape, strides, pads, groups = pat
      inputs = [builder.name(inp), builder.name(weight)]
      if bias is not None: inputs.append(builder.name(bias))
      builder.add_node("ConvTranspose", inputs, [out], kernel_shape=kernel_shape, strides=strides, pads=pads, group=groups)
    elif pat[0] == "BatchNorm":
      _, inp, scale, bias, mean, var, eps = pat
      # ONNX BatchNormalization requires 1-D (C,) params.  The tinygrad buffers
      # may be registered as (1,C,1,1) because _find_buffer_shape picks the first
      # RESHAPE it finds.  Emit explicit Reshape nodes to flatten them.
      bn_params = []
      for param_buf in (scale, bias, mean, var):
        param_name = builder.name(param_buf)
        buf_size = int(param_buf.arg)
        flat_name = builder.fresh_name("bn1d_")
        flat_shape_name = builder.fresh_name("shape_")
        builder.add_const_tensor(flat_shape_name, np.array([buf_size], dtype=np.int64))
        builder.add_node("Reshape", [param_name, flat_shape_name], [flat_name])
        bn_params.append(flat_name)
      builder.add_node("BatchNormalization",
                        [builder.name(inp)] + bn_params, [out], epsilon=eps)
    elif pat[0] == "GlobalAvgPool":
      _, inp = pat
      # GlobalAveragePool produces (N,C,1,1), then we need Reshape to match the root shape
      gap_name = builder.fresh_name("gap_")
      builder.add_node("GlobalAveragePool", [builder.name(inp)], [gap_name])
      out_shape = list(int(x) for x in uop.shape)
      shape_name = builder.fresh_name("shape_")
      builder.add_const_tensor(shape_name, np.array(out_shape, dtype=np.int64))
      builder.add_node("Reshape", [gap_name, shape_name], [out])
    elif pat[0] == "Activation":
      _, inp, onnx_op, attrs = pat
      builder.add_node(onnx_op, [builder.name(inp)], [out], **attrs)
    elif pat[0] == "Pool":
      _, inp, onnx_op, attrs = pat
      builder.add_node(onnx_op, [builder.name(inp)], [out], **attrs)
    elif pat[0] == "Resize":
      _, inp, attrs = pat
      mode = attrs["mode"]
      output_size = attrs["output_size"]
      coord_mode = attrs["coordinate_transformation_mode"]
      # ONNX Resize: inputs = [X, roi, scales, sizes]
      roi_name = builder.fresh_name("roi_")
      builder.add_const_tensor(roi_name, np.array([], dtype=np.float32))
      scales_name = builder.fresh_name("scales_")
      builder.add_const_tensor(scales_name, np.array([], dtype=np.float32))
      sizes_name = builder.fresh_name("sizes_")
      builder.add_const_tensor(sizes_name, np.array(output_size, dtype=np.int64))
      extra_attrs = {"mode": mode, "coordinate_transformation_mode": coord_mode}
      if mode == "nearest": extra_attrs["nearest_mode"] = attrs.get("nearest_mode", "floor")
      builder.add_node("Resize", [builder.name(inp), roi_name, scales_name, sizes_name], [out], **extra_attrs)
    return

  out = builder.name(uop)
  dt = _uop_dtype(uop)

  # skip shape parameters, markers, scheduling ops, and already-handled ops
  if _is_shape_uop(uop): return
  if uop.op in {Ops.UNIQUE, Ops.DEVICE, Ops.BUFFER, Ops.STORE, Ops.AFTER, Ops.THREEFRY}: return

  # passthrough ops
  if uop.op in {Ops.CONTIGUOUS, Ops.CONTIGUOUS_BACKWARD, Ops.DETACH, Ops.COPY}:
    builder.set_name(uop, builder.name(uop.src[0]))
    return

  if uop.op == Ops.CONST:
    val = uop.arg
    np_dt = {dtypes.float32: np.float32, dtypes.float64: np.float64, dtypes.float16: np.float16,
             dtypes.int32: np.int32, dtypes.int64: np.int64, dtypes.int8: np.int8, dtypes.int16: np.int16,
             dtypes.uint8: np.uint8, dtypes.uint16: np.uint16, dtypes.uint32: np.uint32, dtypes.uint64: np.uint64,
             dtypes.bool: np.bool_}.get(dt, np.float32)
    # Handle unsigned integer overflow (e.g., -1 as uint32 → 0xFFFFFFFF)
    if np.issubdtype(np_dt, np.unsignedinteger) and isinstance(val, int) and val < 0:
      info = np.iinfo(np_dt)
      val = val % (info.max + 1)
    builder.add_const_tensor(out, np.array(val, dtype=np_dt).reshape(()))
    return

  if uop.op == Ops.RESHAPE:
    src_name = builder.name(uop.src[0])
    shape = tuple(int(x) if not isinstance(x, int) else x for x in uop.marg)
    shape_name = builder.fresh_name("shape_")
    builder.add_const_tensor(shape_name, np.array(shape, dtype=np.int64))
    builder.add_node("Reshape", [src_name, shape_name], [out])
    return

  if uop.op == Ops.PERMUTE:
    builder.add_node("Transpose", [builder.name(uop.src[0])], [out], perm=list(uop.marg))
    return

  if uop.op == Ops.EXPAND:
    shape = tuple(int(x) if not isinstance(x, int) else x for x in uop.marg)
    shape_name = builder.fresh_name("shape_")
    builder.add_const_tensor(shape_name, np.array(shape, dtype=np.int64))
    builder.add_node("Expand", [builder.name(uop.src[0]), shape_name], [out])
    return

  if uop.op == Ops.PAD:
    pairs = uop.marg
    begins = [int(b) for b, _ in pairs]
    ends = [int(e) for _, e in pairs]
    pads = begins + ends
    pads_name = builder.fresh_name("pads_")
    builder.add_const_tensor(pads_name, np.array(pads, dtype=np.int64))
    builder.add_node("Pad", [builder.name(uop.src[0]), pads_name], [out])
    return

  if uop.op == Ops.SHRINK:
    pairs = uop.marg
    starts = [int(s) for s, _ in pairs]
    ends = [int(e) for _, e in pairs]
    axes = list(range(len(pairs)))
    starts_name = builder.fresh_name("starts_")
    ends_name = builder.fresh_name("ends_")
    axes_name = builder.fresh_name("axes_")
    builder.add_const_tensor(starts_name, np.array(starts, dtype=np.int64))
    builder.add_const_tensor(ends_name, np.array(ends, dtype=np.int64))
    builder.add_const_tensor(axes_name, np.array(axes, dtype=np.int64))
    builder.add_node("Slice", [builder.name(uop.src[0]), starts_name, ends_name, axes_name], [out])
    return

  if uop.op == Ops.FLIP:
    input_shape = uop.src[0].shape
    starts, ends, axes, steps = [], [], [], []
    for i, flip in enumerate(uop.marg):
      if flip:
        starts.append(int(input_shape[i]) - 1)
        ends.append(-(int(input_shape[i]) + 1))
        axes.append(i)
        steps.append(-1)
    if not axes:
      builder.set_name(uop, builder.name(uop.src[0]))
      return
    starts_name = builder.fresh_name("starts_")
    ends_name = builder.fresh_name("ends_")
    axes_name = builder.fresh_name("axes_")
    steps_name = builder.fresh_name("steps_")
    builder.add_const_tensor(starts_name, np.array(starts, dtype=np.int64))
    builder.add_const_tensor(ends_name, np.array(ends, dtype=np.int64))
    builder.add_const_tensor(axes_name, np.array(axes, dtype=np.int64))
    builder.add_const_tensor(steps_name, np.array(steps, dtype=np.int64))
    builder.add_node("Slice", [builder.name(uop.src[0]), starts_name, ends_name, axes_name, steps_name], [out])
    return

  if uop.op == Ops.REDUCE_AXIS:
    reduce_op, axes = uop.arg
    op_map = {Ops.ADD: "ReduceSum", Ops.MAX: "ReduceMax", Ops.MUL: "ReduceProd"}
    if reduce_op not in op_map: raise RuntimeError(f"unsupported reduce op: {reduce_op}")
    axes_name = builder.fresh_name("axes_")
    builder.add_const_tensor(axes_name, np.array(list(axes), dtype=np.int64))
    builder.add_node(op_map[reduce_op], [builder.name(uop.src[0]), axes_name], [out], keepdims=1)
    return

  if uop.op == Ops.CAST:
    target_dt = uop.dtype.base if isinstance(uop.dtype, PtrDType) else uop.dtype
    builder.add_node("Cast", [builder.name(uop.src[0])], [out], to=_get_onnx_dtype(target_dt))
    return

  simple_unary = {Ops.NEG: "Neg", Ops.SIN: "Sin", Ops.SQRT: "Sqrt", Ops.RECIPROCAL: "Reciprocal"}
  if uop.op in simple_unary:
    builder.add_node(simple_unary[uop.op], [builder.name(uop.src[0])], [out])
    return

  simple_binary = {Ops.ADD: "Add", Ops.MUL: "Mul", Ops.SUB: "Sub", Ops.FDIV: "Div", Ops.POW: "Pow", Ops.MOD: "Mod"}
  if uop.op in simple_binary:
    attrs = {}
    if uop.op == Ops.MOD and dtypes.is_float(dt): attrs["fmod"] = 1
    builder.add_node(simple_binary[uop.op], [builder.name(uop.src[0]), builder.name(uop.src[1])], [out], **attrs)
    return

  if uop.op == Ops.MAX:
    builder.add_node("Max", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out])
    return

  if uop.op == Ops.IDIV:
    builder.add_node("Div", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out])
    return

  if uop.op == Ops.CMPLT:
    builder.add_node("Less", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out])
    return
  if uop.op == Ops.CMPEQ:
    builder.add_node("Equal", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out])
    return
  if uop.op == Ops.CMPNE:
    eq_name = builder.fresh_name("eq_")
    builder.add_node("Equal", [builder.name(uop.src[0]), builder.name(uop.src[1])], [eq_name])
    builder.add_node("Not", [eq_name], [out])
    return

  if uop.op == Ops.WHERE:
    cond_name = builder.name(uop.src[0])
    # Always cast condition to bool — ONNX Where requires bool condition,
    # and the condition tensor may have been cast to float by a shared subgraph.
    cast_name = builder.fresh_name("where_bool_")
    builder.add_node("Cast", [cond_name], [cast_name], to=_get_onnx_dtype(dtypes.bool))
    builder.add_node("Where", [cast_name, builder.name(uop.src[1]), builder.name(uop.src[2])], [out])
    return

  if uop.op == Ops.EXP2:
    ln2_name = builder.fresh_name("ln2_")
    mul_name = builder.fresh_name("mul_")
    np_dt = np.float64 if dt == dtypes.float64 else np.float32
    builder.add_const_tensor(ln2_name, np.array(math.log(2), dtype=np_dt).reshape(()))
    builder.add_node("Mul", [builder.name(uop.src[0]), ln2_name], [mul_name])
    builder.add_node("Exp", [mul_name], [out])
    return

  if uop.op == Ops.LOG2:
    log_name = builder.fresh_name("log_")
    inv_ln2_name = builder.fresh_name("inv_ln2_")
    np_dt = np.float64 if dt == dtypes.float64 else np.float32
    builder.add_node("Log", [builder.name(uop.src[0])], [log_name])
    builder.add_const_tensor(inv_ln2_name, np.array(1.0 / math.log(2), dtype=np_dt).reshape(()))
    builder.add_node("Mul", [log_name, inv_ln2_name], [out])
    return

  if uop.op == Ops.TRUNC:
    _, _, TensorProto, _ = _import_onnx()
    int_name = builder.fresh_name("trunc_int_")
    builder.add_node("Cast", [builder.name(uop.src[0])], [int_name], to=TensorProto.INT64)
    builder.add_node("Cast", [int_name], [out], to=_get_onnx_dtype(dt))
    return

  if uop.op == Ops.AND:
    builder.add_node("And" if dt == dtypes.bool else "BitwiseAnd", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out])
    return
  if uop.op == Ops.OR:
    builder.add_node("Or" if dt == dtypes.bool else "BitwiseOr", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out])
    return
  if uop.op == Ops.XOR:
    builder.add_node("Xor" if dt == dtypes.bool else "BitwiseXor", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out])
    return

  if uop.op == Ops.SHL:
    builder.add_node("BitShift", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out], direction="LEFT")
    return
  if uop.op == Ops.SHR:
    builder.add_node("BitShift", [builder.name(uop.src[0]), builder.name(uop.src[1])], [out], direction="RIGHT")
    return

  if uop.op == Ops.MULACC:
    mul_name = builder.fresh_name("mulacc_mul_")
    builder.add_node("Mul", [builder.name(uop.src[0]), builder.name(uop.src[1])], [mul_name])
    builder.add_node("Add", [mul_name, builder.name(uop.src[2])], [out])
    return

  if uop.op == Ops.BITCAST:
    target_dt = uop.dtype.base if isinstance(uop.dtype, PtrDType) else uop.dtype
    builder.add_node("Cast", [builder.name(uop.src[0])], [out], to=_get_onnx_dtype(target_dt))
    return

  raise RuntimeError(f"unsupported UOp for ONNX export: {uop.op}")

def export_onnx(model, *inputs: Tensor, filename: str | None = None) -> Any:
  """Export a tinygrad model to ONNX format.

  Args:
    model: A callable or object with .forward() method.
    *inputs: Example input Tensors (should be realized).
    filename: Optional path to save the ONNX model.

  Returns:
    onnx.ModelProto
  """
  onnx, helper, TensorProto, numpy_helper = _import_onnx()

  # realize inputs if needed
  for inp in inputs:
    if not inp.uop.base.is_realized: inp.realize()

  # run model forward (lazy) to get output UOp graph
  out = model.forward(*inputs) if hasattr(model, "forward") else model(*inputs)
  outputs = [out] if isinstance(out, Tensor) else list(out)

  # identify input buffer UOps
  input_buf_ids: dict[int, int] = {}
  for i, inp in enumerate(inputs):
    base = inp.uop.base
    if base.op == Ops.BUFFER: input_buf_ids[id(base)] = i

  # collect all UOps via toposort from outputs
  all_uops: dict[UOp, None] = {}
  for o in outputs:
    all_uops.update(o.uop.toposort())

  # detect high-level patterns (MatMul, Gemm)
  patterns, consumed = _detect_patterns(all_uops)

  builder = OnnxBuilder()

  # first pass: register BUFFERs as inputs or initializers
  for uop in all_uops:
    if uop.op != Ops.BUFFER: continue
    buf_id = id(uop)
    buf_dtype = uop.dtype.base if isinstance(uop.dtype, PtrDType) else uop.dtype

    shape = _find_buffer_shape(uop, all_uops)

    if buf_id in input_buf_ids:
      idx = input_buf_ids[buf_id]
      name = f"input{idx}"
      builder.set_name(uop, name)
      builder.graph_inputs.append(helper.make_tensor_value_info(name, _get_onnx_dtype(buf_dtype), list(shape)))
    elif uop.realized is not None:
      name = builder.name(uop)
      np_data = uop.buffer.numpy().reshape(shape)
      builder.add_initializer(name, np_data)
    else:
      name = builder.name(uop)
      builder.graph_inputs.append(helper.make_tensor_value_info(name, _get_onnx_dtype(buf_dtype), list(shape)))

  # second pass: emit ONNX nodes
  for uop in all_uops:
    _emit_uop(builder, uop, patterns, consumed)

  # register outputs
  for i, o in enumerate(outputs):
    out_name = builder.name(o.uop)
    oshape = list(int(x) if not isinstance(x, int) else x for x in o.uop.shape)
    out_dtype = _get_onnx_dtype(_uop_dtype(o.uop))
    builder.graph_outputs.append(helper.make_tensor_value_info(f"output{i}", out_dtype, oshape))
    if out_name != f"output{i}":
      builder.add_node("Identity", [out_name], [f"output{i}"])

  graph = helper.make_graph(builder.nodes, "tinygrad_export", builder.graph_inputs, builder.graph_outputs, builder.initializers)
  model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
  model_proto.ir_version = 9

  # Dead code elimination: remove nodes and initializers not reachable from outputs.
  # tinygrad's UOp graph often contains im2col chains that become dead after pattern
  # matching replaces them with high-level Conv/BN/Pool ops.
  _eliminate_dead_code(model_proto)

  onnx.checker.check_model(model_proto)

  # Validate structure: warn about unmatched matmul patterns that ORT would
  # materialize as full expanded tensors (Expand -> Mul -> ReduceSum instead of MatMul).
  _warn_unmatched_matmuls(model_proto, numpy_helper)

  if filename is not None: onnx.save(model_proto, filename)
  return model_proto


def _eliminate_dead_code(model_proto) -> None:
  """Remove ONNX nodes and initializers not reachable from the graph outputs."""
  graph = model_proto.graph

  # Build map: output_name → node
  node_by_output: dict[str, Any] = {}
  for node in graph.node:
    for out_name in node.output:
      node_by_output[out_name] = node

  # BFS backward from graph outputs to find all needed names
  needed_names: set[str] = set()
  needed_node_ids: set[int] = set()
  queue = [o.name for o in graph.output]
  while queue:
    name = queue.pop()
    if name in needed_names:
      continue
    needed_names.add(name)
    if name in node_by_output:
      node = node_by_output[name]
      nid = id(node)
      if nid not in needed_node_ids:
        needed_node_ids.add(nid)
        for inp_name in node.input:
          if inp_name and inp_name not in needed_names:
            queue.append(inp_name)

  # Filter nodes
  alive_nodes = [n for n in graph.node if id(n) in needed_node_ids]
  del graph.node[:]
  graph.node.extend(alive_nodes)

  # Filter initializers
  alive_inits = [i for i in graph.initializer if i.name in needed_names]
  del graph.initializer[:]
  graph.initializer.extend(alive_inits)

  # Keep all graph inputs — some ORT versions require initializers to be listed as inputs
  # Only remove inputs that are truly dead (not needed and not an initializer)
  init_names = {i.name for i in graph.initializer}
  alive_inputs = [i for i in graph.input if i.name in needed_names or i.name in init_names]
  del graph.input[:]
  graph.input.extend(alive_inputs)

def _warn_unmatched_matmuls(model_proto, numpy_helper) -> None:
  """Check for Expand -> Mul -> ReduceSum chains that should be MatMul ops.

  When the pattern matcher fails to fuse a matmul, the exporter emits it as
  element-wise Expand + Mul + ReduceSum.  ORT materializes the full expanded
  tensors, which can easily exceed available memory (e.g. 782 GB for a ViT
  attention layer at batch_size=384).
  """
  import warnings
  graph = model_proto.graph

  node_by_output: dict[str, Any] = {}
  for node in graph.node:
    for out in node.output:
      node_by_output[out] = node
  consumers: dict[str, list] = {}
  for node in graph.node:
    for inp in node.input:
      consumers.setdefault(inp, []).append(node)

  init_values: dict[str, np.ndarray] = {}
  for init in graph.initializer:
    init_values[init.name] = numpy_helper.to_array(init)

  total_bytes = 0
  count = 0
  for node in graph.node:
    if node.op_type != "Mul": continue
    n_expand = 0
    max_elems = 0
    for inp in node.input:
      if inp in node_by_output and node_by_output[inp].op_type == "Expand":
        en = node_by_output[inp]
        if en.input[1] in init_values:
          shape = init_values[en.input[1]].astype(int)
          max_elems = max(max_elems, int(np.prod(shape)))
          n_expand += 1
    if n_expand < 2: continue
    has_reduce = any(c.op_type == "ReduceSum" for c in consumers.get(node.output[0], []))
    if has_reduce:
      count += 1
      total_bytes += max_elems * 4 * 2  # 2 expanded fp32 tensors

  if count > 0:
    warnings.warn(
      f"ONNX export has {count} unmatched matmul pattern(s) "
      f"(Expand→Mul→ReduceSum instead of MatMul). "
      f"ORT will try to allocate ~{total_bytes / 1024**3:.1f} GB of expanded tensors. "
      f"This usually means the pattern matcher in export_onnx._detect_patterns "
      f"failed to fuse these into MatMul ops.",
      stacklevel=3,
    )

def _find_buffer_shape(buf_uop: UOp, all_uops: dict[UOp, None]) -> tuple[int, ...]:
  buf_size = int(buf_uop.arg)
  for uop in all_uops:
    if uop.op == Ops.RESHAPE and uop.src[0] is buf_uop:
      shape = tuple(int(x) if not isinstance(x, int) else x for x in uop.marg)
      # Only return this shape if total elements match the buffer size
      total = 1
      for s in shape: total *= s
      if total == buf_size: return shape
  return (buf_size,)
