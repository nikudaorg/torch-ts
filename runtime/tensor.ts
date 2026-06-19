import type { DType } from '../types/basic/dtype.ts';
import type { Device } from '../types/basic/device.ts';
import type { Shape } from '../types/basic/shape.ts';
import type { Tensor } from '../types/basic/tensor.ts';
import type { TensorReference, TensorResult } from './protocol.ts';

const handleSymbol = Symbol('torch-ts.tensor-handle');
const inspectSymbol = Symbol.for('nodejs.util.inspect.custom');

type ManagedTensor = Tensor & {
  readonly [handleSymbol]: number;
  readonly [inspectSymbol]: () => string;
};

export function createTensor(
  result: TensorResult,
  release?: (handle: number) => void
): Tensor {
  const tensor = {
    kind: 'Tensor',
    shape: Object.freeze([...result.shape]) as unknown as Shape,
    dtype: result.dtype as DType,
    device: result.device as Device,
    toString: () => result.repr
  } as ManagedTensor;

  Object.defineProperties(tensor, {
    [handleSymbol]: {
      value: result.handle,
      enumerable: false,
      writable: false
    },
    [inspectSymbol]: {
      value: () => result.repr,
      enumerable: false,
      writable: false
    }
  });
  releaseRegistry.register(tensor, {
    handle: result.handle,
    ...(release === undefined ? {} : { release })
  });

  return Object.freeze(tensor);
}

export function isManagedTensor(value: unknown): value is ManagedTensor {
  return (
    typeof value === 'object' &&
    value !== null &&
    handleSymbol in value &&
    (value as { kind?: unknown }).kind === 'Tensor'
  );
}

export function tensorReference(tensor: ManagedTensor): TensorReference {
  return { $tensor: tensor[handleSymbol] };
}

export function isTensorResult(value: unknown): value is TensorResult {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as Partial<TensorResult>;
  return (
    candidate.kind === 'tensor' &&
    typeof candidate.handle === 'number' &&
    Array.isArray(candidate.shape) &&
    typeof candidate.dtype === 'string' &&
    typeof candidate.device === 'string' &&
    typeof candidate.repr === 'string'
  );
}

const releaseRegistry = new FinalizationRegistry<{
  handle: number;
  release?: (handle: number) => void;
}>(({ handle, release }) => {
  release?.(handle);
});
