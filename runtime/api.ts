import { fileURLToPath } from 'node:url';
import type {
  API,
  DefaultableDType
} from '../types/api/main.ts';
import type { Device } from '../types/basic/device.ts';
import type { Tensor } from '../types/basic/tensor.ts';
import type { JsonValue } from './protocol.ts';
import { PythonRpcClient } from './rpc.ts';
import {
  createTensor,
  isManagedTensor,
  isTensorResult,
  tensorReference
} from './tensor.ts';

export interface CreateAPIOptions<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device
> {
  defaultDType: DefaultDType;
  defaultDevice: DefaultDevice;
  pythonPath?: string;
  workerPath?: string;
  endpoint?: string;
  startupTimeoutMs?: number;
  env?: NodeJS.ProcessEnv;
}

export interface RuntimeControl {
  close(): Promise<void>;
}

export type AsyncAPI<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device
> = API<DefaultDType, DefaultDevice, 'async'> & RuntimeControl;

type RuntimeMethod = (...args: never[]) => Promise<Tensor>;

export function createAPI<
  const DefaultDType extends DefaultableDType,
  const DefaultDevice extends Device
>(
  options: CreateAPIOptions<DefaultDType, DefaultDevice>
): AsyncAPI<DefaultDType, DefaultDevice> {
  const workerPath =
    options.workerPath ??
    fileURLToPath(new URL('../python/worker.py', import.meta.url));
  const client = new PythonRpcClient({
    pythonPath: options.pythonPath ?? defaultPythonPath(),
    workerPath,
    startupTimeoutMs: options.startupTimeoutMs ?? 30_000,
    ...(options.endpoint === undefined ? {} : { endpoint: options.endpoint }),
    ...(options.env === undefined ? {} : { env: options.env })
  });
  const initialized = client.call('__init__', [], {
    default_dtype: options.defaultDType,
    default_device: options.defaultDevice
  });
  void initialized.catch(async () => {
    try {
      await client.close();
    } catch {
      // Preserve the initialization error as the operation-facing failure.
    }
  });

  const invoke = async (
    op: string,
    args: unknown[] = [],
    kwargs: Record<string, unknown> = {}
  ): Promise<Tensor> => {
    await initialized;
    const result = await client.call(
      op,
      encodeArray(args),
      encodeRecord(kwargs)
    );
    if (!isTensorResult(result)) {
      throw new TypeError(`Torch operation ${op} did not return a tensor`);
    }
    return createTensor(result, handle => {
      void client.call('__release__', [handle]).catch(() => {
        // The worker may already be closing when finalization runs.
      });
    });
  };

  const unary = (op: string): RuntimeMethod =>
    ((input: Tensor) => invoke(op, [input])) as RuntimeMethod;

  const api = {
    close: () => client.close(),

    zeros: (shape: number[], dtype?: string, device?: string) =>
      invoke('zeros', [shape], defined({ dtype, device })),
    ones: (shape: number[], dtype?: string, device?: string) =>
      invoke('ones', [shape], defined({ dtype, device })),
    rand: (shape: number[], dtype?: string, device?: string) =>
      invoke('rand', [shape], defined({ dtype, device })),
    randn: (shape: number[], dtype?: string, device?: string) =>
      invoke('randn', [shape], defined({ dtype, device })),
    full: (
      shape: number[],
      fillValue: unknown,
      dtype?: string,
      device?: string
    ) =>
      invoke(
        'full',
        [shape, fillValue],
        defined({ dtype, device })
      ),
    zeros_like: (input: Tensor, dtype?: string, device?: string) =>
      invoke('zeros_like', [input], defined({ dtype, device })),
    ones_like: (input: Tensor, dtype?: string, device?: string) =>
      invoke('ones_like', [input], defined({ dtype, device })),
    arange: (
      start: number,
      end: number,
      step?: number,
      dtype?: string,
      device?: string
    ) =>
      invoke(
        'arange',
        step === undefined ? [start, end] : [start, end, step],
        defined({ dtype, device })
      ),

    add: (input: Tensor, other: Tensor, alpha?: unknown) =>
      invoke('add', [input, other], defined({ alpha })),
    sub: (input: Tensor, other: Tensor, alpha?: unknown) =>
      invoke('sub', [input, other], defined({ alpha })),
    mul: (input: Tensor, other: Tensor) => invoke('mul', [input, other]),

    sum: reducer('sum', invoke),
    mean: reducer('mean', invoke),
    prod: reducer('prod', invoke),
    nansum: reducer('nansum', invoke),
    nanprod: reducer('nanprod', invoke),

    unsqueeze: (input: Tensor, dim: number) =>
      invoke('unsqueeze', [input, dim]),
    squeeze: (input: Tensor, dim?: number | number[]) =>
      invoke('squeeze', [input], defined({ dim })),
    transpose: (input: Tensor, dim0: number, dim1: number) =>
      invoke('transpose', [input, dim0, dim1]),
    cat: (tensors: Tensor[], dim?: number) =>
      invoke('cat', [tensors], defined({ dim })),
    stack: (tensors: Tensor[], dim?: number) =>
      invoke('stack', [tensors], defined({ dim })),

    abs: unary('abs'),
    neg: unary('neg'),
    sign: unary('sign'),
    sgn: unary('sgn'),
    reciprocal: unary('reciprocal'),
    square: unary('square'),
    sqrt: unary('sqrt'),
    rsqrt: unary('rsqrt'),
    exp: unary('exp'),
    exp2: unary('exp2'),
    expm1: unary('expm1'),
    log: unary('log'),
    log2: unary('log2'),
    log10: unary('log10'),
    log1p: unary('log1p'),
    logit: (input: Tensor, eps?: number | null) =>
      invoke('logit', [input], defined({ eps })),
    sigmoid: unary('sigmoid'),
    sin: unary('sin'),
    cos: unary('cos'),
    tan: unary('tan'),
    asin: unary('asin'),
    acos: unary('acos'),
    atan: unary('atan'),
    sinh: unary('sinh'),
    cosh: unary('cosh'),
    tanh: unary('tanh'),
    asinh: unary('asinh'),
    acosh: unary('acosh'),
    atanh: unary('atanh'),
    ceil: unary('ceil'),
    floor: unary('floor'),
    round: (input: Tensor, decimals?: number) =>
      invoke('round', [input], defined({ decimals })),
    trunc: unary('trunc'),
    frac: unary('frac'),
    logical_not: unary('logical_not'),
    bitwise_not: unary('bitwise_not'),
    isnan: unary('isnan'),
    isinf: unary('isinf'),
    isfinite: unary('isfinite'),
    isreal: unary('isreal'),
    isposinf: unary('isposinf'),
    isneginf: unary('isneginf'),
    angle: unary('angle'),
    conj: unary('conj'),
    real: unary('real'),
    imag: unary('imag'),
    erf: unary('erf'),
    erfc: unary('erfc'),
    erfinv: unary('erfinv'),
    lgamma: unary('lgamma'),
    digamma: unary('digamma'),
    sinc: unary('sinc'),
    i0: unary('i0')
  };

  return api as unknown as AsyncAPI<DefaultDType, DefaultDevice>;
}

function reducer(
  op: string,
  invoke: (
    op: string,
    args?: unknown[],
    kwargs?: Record<string, unknown>
  ) => Promise<Tensor>
): RuntimeMethod {
  return ((
    input: Tensor,
    dim?: number | number[],
    keepdim?: boolean,
    dtype?: string
  ) => invoke(op, [input], defined({ dim, keepdim, dtype }))) as RuntimeMethod;
}

function defined(
  values: Record<string, unknown>
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(values).filter(([, value]) => value !== undefined)
  );
}

function encodeArray(values: unknown[]): JsonValue[] {
  return values.map(encodeValue);
}

function encodeRecord(
  values: Record<string, unknown>
): Record<string, JsonValue> {
  return Object.fromEntries(
    Object.entries(values).map(([key, value]) => [key, encodeValue(value)])
  );
}

function encodeValue(value: unknown): JsonValue {
  if (isManagedTensor(value)) return tensorReference(value);
  if (value === null) return null;
  if (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return value;
  }
  if (Array.isArray(value)) return value.map(encodeValue);
  if (typeof value === 'object' && value !== null) {
    return encodeRecord(value as Record<string, unknown>);
  }
  throw new TypeError(`Cannot send ${typeof value} to the torch runtime`);
}

function defaultPythonPath(): string {
  return process.platform === 'win32' ? 'python' : 'python3';
}
