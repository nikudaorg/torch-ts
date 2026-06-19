import { API } from './types/api';

declare const t: API<'float32', 'cpu', 'sync'>;

// 1. Basic concatenation along the first dimension.
// Expected: shape [6, 4], dtype float32, device cpu.
const cat01 = t.cat(
  [t.zeros([3, 4], 'float32', 'cpu'), t.zeros([3, 4], 'float32', 'cpu')],
  0
);

// 2. Basic concatenation along the second dimension.
// Expected: shape [3, 8], dtype float32, device cpu.
const cat02 = t.cat(
  [t.zeros([3, 4], 'float32', 'cpu'), t.zeros([3, 4], 'float32', 'cpu')],
  1
);

// 3. Input sizes may differ along the concatenation dimension.
// Expected: shape [10, 4], dtype float32, device cpu.
const cat03 = t.cat(
  [
    t.zeros([2, 4], 'float32', 'cpu'),
    t.zeros([3, 4], 'float32', 'cpu'),
    t.zeros([5, 4], 'float32', 'cpu')
  ],
  0
);

// 4. Negative dimensions are normalized relative to the tensor rank.
// dim -1 is dim 1 for these rank-2 tensors.
// Expected: shape [2, 9], dtype float32, device cpu.
const cat04 = t.cat(
  [
    t.zeros([2, 3], 'float32', 'cpu'),
    t.zeros([2, 5], 'float32', 'cpu'),
    t.zeros([2, 1], 'float32', 'cpu')
  ],
  -1
);

// 5. Concatenation of rank-3 tensors along an interior dimension.
// Expected: shape [2, 8, 4], dtype float32, device cpu.
const cat05 = t.cat(
  [t.zeros([2, 3, 4], 'float32', 'cpu'), t.zeros([2, 5, 4], 'float32', 'cpu')],
  1
);

// 6. PyTorch's special case: a one-dimensional empty tensor [0]
// may be concatenated with a non-empty tensor of another rank.
// Expected: shape [2, 3], dtype float32, device cpu.
const cat06 = t.cat(
  [t.zeros([0], 'float32', 'cpu'), t.zeros([2, 3], 'float32', 'cpu')],
  0
);

// 7. Floating-point promotion: float32 + float64 -> float64.
// Expected: shape [4, 3], dtype float64, device cpu.
const cat07 = t.cat(
  [t.zeros([2, 3], 'float32', 'cpu'), t.zeros([2, 3], 'float64', 'cpu')],
  0
);

// 8. Integer promotion: int32 + int64 -> int64.
// Expected: shape [2, 6], dtype int64, device cpu.
const cat08 = t.cat(
  [t.zeros([2, 3], 'int32', 'cpu'), t.zeros([2, 3], 'int64', 'cpu')],
  1
);

// 9. Integer and floating-point promotion: int64 + float32 -> float32.
// This is intentional PyTorch behavior; it does not promote to float64.
// Expected: shape [4, 3], dtype float32, device cpu.
const cat09 = t.cat(
  [t.zeros([2, 3], 'int64', 'cpu'), t.zeros([2, 3], 'float32', 'cpu')],
  0
);

// 10. Signed/unsigned integer promotion: uint8 + int8 -> int16.
// Expected: shape [4, 3], dtype int16, device cpu.
const cat10 = t.cat(
  [t.zeros([2, 3], 'uint8', 'cpu'), t.zeros([2, 3], 'int8', 'cpu')],
  0
);

// 11. Matching non-CPU device strings should be preserved.
// Expected: shape [4, 3], dtype float16, device cuda:0.
const cat11 = t.cat(
  [t.zeros([2, 3], 'float16', 'cuda:0'), t.zeros([2, 3], 'float16', 'cuda:0')],
  0
);

// 12. Sizes differ outside the concatenation dimension.
// Expected warning: tensors must have matching sizes outside dim 0;
// dimension 1 is 4 in the first tensor but 5 in the second tensor.
const cat12 = t.cat(
  [t.zeros([2, 4], 'float32', 'cpu'), t.zeros([3, 5], 'float32', 'cpu')],
  0
);

// 13. Non-empty tensors have different ranks.
// Expected warning: tensors must have the same number of dimensions;
// the inputs have ranks 2 and 1.
const cat13 = t.cat(
  [t.zeros([2, 3], 'float32', 'cpu'), t.zeros([6], 'float32', 'cpu')],
  0
);

// 14. Concatenation dimension is outside the valid rank-2 range.
// Valid dimensions are -2, -1, 0, and 1.
// Expected warning: dimension 2 is out of range for rank-2 tensors.
const cat14 = t.cat(
  [t.zeros([2, 3], 'float32', 'cpu'), t.zeros([2, 3], 'float32', 'cpu')],
  2
);

// 15. Devices do not match.
// Expected warning: all tensors must be on the same device;
// the inputs are on cpu and cuda:0.
const cat15 = t.cat(
  [t.zeros([2, 3], 'float32', 'cpu'), t.zeros([2, 3], 'float32', 'cuda:0')],
  0
);

// 1. Successive integer widening.
// Expected: shape [6, 4], dtype int32, device cpu.
const multiCat01 = t.cat(
  [
    t.zeros([1, 4], 'int8', 'cpu'),
    t.zeros([2, 4], 'int16', 'cpu'),
    t.zeros([3, 4], 'int32', 'cpu')
  ],
  0
);

// 2. Signed and unsigned promotion must consider all three inputs.
// uint8 + int8 promotes to int16; the existing int16 input preserves that result.
// Expected: shape [2, 6], dtype int16, device cpu.
const multiCat02 = t.cat(
  [
    t.zeros([2, 1], 'uint8', 'cpu'),
    t.zeros([2, 2], 'int8', 'cpu'),
    t.zeros([2, 3], 'int16', 'cpu')
  ],
  1
);

// 3. bool does not prevent signed/unsigned integer promotion.
// bool + uint8 + int8 -> int16.
// Expected: shape [6, 3], dtype int16, device cpu.
const multiCat03 = t.cat(
  [
    t.zeros([1, 3], 'bool', 'cpu'),
    t.zeros([2, 3], 'uint8', 'cpu'),
    t.zeros([3, 3], 'int8', 'cpu')
  ],
  0
);

// 4. The widest integral input determines the integral result.
// Expected: shape [3, 5], dtype int64, device cpu.
const multiCat04 = t.cat(
  [
    t.zeros([3, 1], 'bool', 'cpu'),
    t.zeros([3, 2], 'int16', 'cpu'),
    t.zeros([3, 2], 'int64', 'cpu')
  ],
  1
);

// 5. A floating category wins over an integral category even when the
// floating dtype has fewer bits than the integer dtype.
// bool + int64 + float16 -> float16, not float32 or float64.
// Expected: shape [6, 2], dtype float16, device cpu.
const multiCat05 = t.cat(
  [
    t.zeros([1, 2], 'bool', 'cpu'),
    t.zeros([2, 2], 'int64', 'cpu'),
    t.zeros([3, 2], 'float16', 'cpu')
  ],
  0
);

// 6. The third input must be allowed to widen an already promoted result.
// int64 + float16 initially gives float16; the final float32 widens it.
// Expected: shape [2, 6], dtype float32, device cpu.
const multiCat06 = t.cat(
  [
    t.zeros([2, 1], 'int64', 'cpu'),
    t.zeros([2, 2], 'float16', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu')
  ],
  1
);

// 7. float16 and bfloat16 together promote to float32.
// Expected: shape [6, 4], dtype float32, device cpu.
const multiCat07 = t.cat(
  [
    t.zeros([1, 4], 'float16', 'cpu'),
    t.zeros([2, 4], 'bfloat16', 'cpu'),
    t.zeros([3, 4], 'float32', 'cpu')
  ],
  0
);

// 8. A later float64 input must dominate float16, bfloat16, and float32.
// Expected: shape [2, 7], dtype float64, device cpu.
const multiCat08 = t.cat(
  [
    t.zeros([2, 1], 'float16', 'cpu'),
    t.zeros([2, 2], 'bfloat16', 'cpu'),
    t.zeros([2, 4], 'float64', 'cpu')
  ],
  1
);

// 9. Integral inputs do not force float64, but an actual float64 input does.
// Expected: shape [6, 3], dtype float64, device cpu.
const multiCat09 = t.cat(
  [
    t.zeros([1, 3], 'int64', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu'),
    t.zeros([3, 3], 'float64', 'cpu')
  ],
  0
);

// 10. A complex category wins over real and integral categories.
// int64 + float32 + complex64 -> complex64.
// Expected: shape [3, 6], dtype complex64, device cpu.
const multiCat10 = t.cat(
  [
    t.zeros([3, 1], 'int64', 'cpu'),
    t.zeros([3, 2], 'float32', 'cpu'),
    t.zeros([3, 3], 'complex64', 'cpu')
  ],
  1
);

// 11. float64 combined with complex64 requires complex128 because the
// complex result must preserve float64 component precision.
// Expected: shape [6, 2], dtype complex128, device cpu.
const multiCat11 = t.cat(
  [
    t.zeros([1, 2], 'int64', 'cpu'),
    t.zeros([2, 2], 'float64', 'cpu'),
    t.zeros([3, 2], 'complex64', 'cpu')
  ],
  0
);

// 12. Four inputs spanning every major dtype category.
// bool + int16 + float32 + complex64 -> complex64.
// Expected: shape [10, 2], dtype complex64, device cpu.
const multiCat12 = t.cat(
  [
    t.zeros([1, 2], 'bool', 'cpu'),
    t.zeros([2, 2], 'int16', 'cpu'),
    t.zeros([3, 2], 'float32', 'cpu'),
    t.zeros([4, 2], 'complex64', 'cpu')
  ],
  0
);

// 13. An empty [0] tensor contributes no elements to the resulting shape,
// but its dtype still participates in promotion.
// float32 + float64(empty) + float16 -> float64.
// Expected: shape [6, 3], dtype float64, device cpu.
const multiCat13 = t.cat(
  [
    t.zeros([2, 3], 'float32', 'cpu'),
    t.zeros([0], 'float64', 'cpu'),
    t.zeros([4, 3], 'float16', 'cpu')
  ],
  0
);

// 14. First ordering for a permutation-invariance check.
// Expected: shape [6, 3], dtype complex64, device cpu.
const multiCat14 = t.cat(
  [
    t.zeros([1, 3], 'int64', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu'),
    t.zeros([3, 3], 'complex64', 'cpu')
  ],
  0
);

// 15. Same dtype set as multiCat14, but in a different order.
// The inferred dtype must not depend on input ordering.
// Expected: shape [6, 3], dtype complex64, device cpu.
const multiCat15 = t.cat(
  [
    t.zeros([3, 3], 'complex64', 'cpu'),
    t.zeros([1, 3], 'int64', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu')
  ],
  0
);

// 1. Insert a new leading dimension containing the two inputs.
// Expected: shape [2, 3, 4], dtype float32, device cpu.
const stack01 = t.stack(
  [t.zeros([3, 4], 'float32', 'cpu'), t.zeros([3, 4], 'float32', 'cpu')],
  0
);

// 2. Insert the new dimension between the existing dimensions.
// There are three input tensors, so the inserted dimension has size 3.
// Expected: shape [3, 3, 4], dtype float32, device cpu.
const stack02 = t.stack(
  [
    t.zeros([3, 4], 'float32', 'cpu'),
    t.zeros([3, 4], 'float32', 'cpu'),
    t.zeros([3, 4], 'float32', 'cpu')
  ],
  1
);

// 3. Insert the new dimension after all existing dimensions.
// Expected: shape [3, 4, 2], dtype float32, device cpu.
const stack03 = t.stack(
  [t.zeros([3, 4], 'float32', 'cpu'), t.zeros([3, 4], 'float32', 'cpu')],
  2
);

// 4. For rank-2 inputs, dim -1 is equivalent to dim 2.
// Expected: shape [3, 4, 3], dtype float32, device cpu.
const stack04 = t.stack(
  [
    t.zeros([3, 4], 'float32', 'cpu'),
    t.zeros([3, 4], 'float32', 'cpu'),
    t.zeros([3, 4], 'float32', 'cpu')
  ],
  -1
);

// 5. Unlike cat, stack supports zero-dimensional scalar tensors.
// Stacking three scalars creates a one-dimensional tensor.
// Expected: shape [3], dtype float32, device cpu.
const stack05 = t.stack(
  [
    t.zeros([], 'float32', 'cpu'),
    t.zeros([], 'float32', 'cpu'),
    t.zeros([], 'float32', 'cpu')
  ],
  0
);

// 6. Empty tensors may be stacked when their complete shapes match.
// Expected: shape [2, 0, 3], dtype float32, device cpu.
const stack06 = t.stack(
  [t.zeros([0, 3], 'float32', 'cpu'), t.zeros([0, 3], 'float32', 'cpu')],
  0
);

// 7. Empty tensors still participate in dtype promotion.
// float32 + float64 -> float64.
// Expected: shape [0, 2, 3], dtype float64, device cpu.
const stack07 = t.stack(
  [t.zeros([0, 3], 'float32', 'cpu'), t.zeros([0, 3], 'float64', 'cpu')],
  1
);

// 8. Promotion must include every input, not only the first two.
// int8 + int16 + int32 -> int32.
// Expected: shape [3, 2, 3], dtype int32, device cpu.
const stack08 = t.stack(
  [
    t.zeros([2, 3], 'int8', 'cpu'),
    t.zeros([2, 3], 'int16', 'cpu'),
    t.zeros([2, 3], 'int32', 'cpu')
  ],
  0
);

// 9. bool + uint8 + int8 -> int16.
// Signed and unsigned 8-bit integers require a wider signed result.
// Expected: shape [2, 3, 3], dtype int16, device cpu.
const stack09 = t.stack(
  [
    t.zeros([2, 3], 'bool', 'cpu'),
    t.zeros([2, 3], 'uint8', 'cpu'),
    t.zeros([2, 3], 'int8', 'cpu')
  ],
  1
);

// 10. int64 + float16 initially promotes to float16, but the third
// float32 input widens the final result to float32.
// Expected: shape [3, 2, 3], dtype float32, device cpu.
const stack10 = t.stack(
  [
    t.zeros([2, 3], 'int64', 'cpu'),
    t.zeros([2, 3], 'float16', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu')
  ],
  0
);

// 11. float16 and bfloat16 together promote to float32.
// Expected: shape [2, 3, 3], dtype float32, device cpu.
const stack11 = t.stack(
  [
    t.zeros([2, 3], 'float16', 'cpu'),
    t.zeros([2, 3], 'bfloat16', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu')
  ],
  2
);

// 12. float64 combined with complex64 requires complex128.
// The int64 input does not change that final result.
// Expected: shape [3, 2, 3], dtype complex128, device cpu.
const stack12 = t.stack(
  [
    t.zeros([2, 3], 'int64', 'cpu'),
    t.zeros([2, 3], 'float64', 'cpu'),
    t.zeros([2, 3], 'complex64', 'cpu')
  ],
  0
);

// 13. First ordering for a permutation-invariance test.
// int64 + float32 + complex64 -> complex64.
// Expected: shape [3, 2, 3], dtype complex64, device cpu.
const stack13 = t.stack(
  [
    t.zeros([2, 3], 'int64', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu'),
    t.zeros([2, 3], 'complex64', 'cpu')
  ],
  0
);

// 14. Same dtype set as stack13, in a different order.
// Dtype inference must not depend on input ordering.
// Expected: shape [3, 2, 3], dtype complex64, device cpu.
const stack14 = t.stack(
  [
    t.zeros([2, 3], 'complex64', 'cpu'),
    t.zeros([2, 3], 'int64', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu')
  ],
  0
);

// 15. Stack requires complete shape equality.
// Matching rank and one matching dimension are insufficient.
// Expected warning: stack expects every tensor to have the same shape;
// got [2, 3] at input 0 and [2, 4] at input 1.
// PyTorch raises RuntimeError.
const stack15 = t.stack(
  [t.zeros([2, 3], 'float32', 'cpu'), t.zeros([2, 4], 'float32', 'cpu')],
  0
);

// 16. Cat's special permission for a one-dimensional empty tensor [0]
// does not apply to stack.
// Expected warning: stack expects every tensor to have the same shape;
// got [0] at input 0 and [2, 3] at input 1.
// PyTorch raises RuntimeError.
const stack16 = t.stack(
  [t.zeros([0], 'float32', 'cpu'), t.zeros([2, 3], 'float32', 'cpu')],
  0
);

// 17. Rank-2 inputs produce rank-3 output, so valid dimensions are
// -3, -2, -1, 0, 1, and 2.
// Expected warning: dimension 3 is out of range;
// expected a dimension in the range [-3, 2].
// PyTorch raises IndexError.
const stack17 = t.stack(
  [t.zeros([2, 3], 'float32', 'cpu'), t.zeros([2, 3], 'float32', 'cpu')],
  3
);

// 18. All input tensors must be on exactly the same device.
// Expected warning: all tensors must be on the same device;
// got cpu at input 0 and cuda:0 at input 1.
// PyTorch raises RuntimeError.
const stack18 = t.stack(
  [t.zeros([2, 3], 'float32', 'cpu'), t.zeros([2, 3], 'float32', 'cuda:0')],
  0
);

// uint16 is a PyTorch shell dtype and cannot be promoted with int16.
// The third float32 tensor does not rescue the operation: promotion fails
// as soon as PyTorch tries to combine uint16 and int16.
//
// Expected warning: these input dtypes do not have a supported common dtype;
// promotion of uint16 and int16 is not supported.
// PyTorch raises RuntimeError:
// "Promotion for uint16, uint32, uint64 types is not supported,
// attempted to promote UInt16 and Short".
const stack19 = t.stack(
  [
    t.zeros([2, 3], 'uint16', 'cpu'),
    t.zeros([2, 3], 'int16', 'cpu'),
    t.zeros([2, 3], 'float32', 'cpu')
  ],
  0
);
