import { Device } from '../../basic/device';
import { DTypeRealOf, GetDType } from '../../basic/dtype';
import { Shape } from '../../basic/shape';
import { Tensor } from '../../basic/tensor';
import { DefaultableDType, MaybeAsync } from '../main';

export interface ElementWiseUnaryAPI<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device,
  SyncType extends 'sync' | 'async'
> {
  abs: <
    TShape extends Shape,
    TDType extends GetDType<'IFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<
    Tensor<
      TShape,
      TDType extends GetDType<'IF'> ? TDType : DTypeRealOf<TDType>,
      TDevice
    >,
    SyncType
  >;

  neg: <TTensor extends Tensor<Shape, GetDType<'IFCH'>>>(
    input: TTensor
  ) => MaybeAsync<TTensor, SyncType>;

  sign: <TTensor extends Tensor<Shape, GetDType<'BIF'>>>(
    input: TTensor
  ) => MaybeAsync<TTensor, SyncType>;

  sgn: <TTensor extends Tensor<Shape, GetDType<'BIFCH'>>>(
    input: TTensor
  ) => MaybeAsync<TTensor, SyncType>;

  reciprocal: <
    TShape extends Shape,
    TDType extends GetDType<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<
    Tensor<
      TShape,
      TDType extends GetDType<'BI'> ? DefaultDType : TDType,
      TDevice
    >,
    SyncType
  >;

  square: <
    TShape extends Shape,
    TDType extends GetDType<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<
    Tensor<TShape, TDType extends GetDType<'B'> ? 'int64' : TDType, TDevice>,
    SyncType
  >;

  sqrt: this['reciprocal'];

  rsqrt: this['sqrt'];

  exp: this['sqrt'];

  exp2: this['sqrt'];

  expm1: this['sqrt'];

  log: this['sqrt'];

  log2: this['sqrt'];

  log10: this['sqrt'];

  log1p: this['sqrt'];

  logit: <
    TShape extends Shape,
    TDType extends GetDType<'BIF'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>,
    eps?: number | null
  ) => MaybeAsync<
    Tensor<
      TShape,
      TDType extends GetDType<'BI'> ? DefaultDType : TDType,
      TDevice
    >,
    SyncType
  >;

  sigmoid: this['sqrt'];

  sin: this['sqrt'];

  cos: this['sqrt'];

  tan: this['sqrt'];

  asin: this['sqrt'];

  acos: this['sqrt'];

  atan: this['sqrt'];

  sinh: this['sqrt'];

  cosh: this['sqrt'];

  tanh: this['sqrt'];

  asinh: this['sqrt'];

  acosh: this['sqrt'];

  atanh: this['sqrt'];

  ceil: <TTensor extends Tensor<Shape, GetDType<'IF'>>>(
    input: TTensor
  ) => MaybeAsync<TTensor, SyncType>;

  floor: this['ceil'];

  round: <
    TShape extends Shape,
    TDevice extends Device,
    TDecimals extends number | undefined = undefined,
    TDType extends [TDecimals] extends [undefined]
      ? GetDType<'IF'>
      : GetDType<'F'> = [TDecimals] extends [undefined]
      ? GetDType<'IF'>
      : GetDType<'F'>
  >(
    input: Tensor<TShape, TDType, TDevice>,
    decimals?: TDecimals
  ) => MaybeAsync<Tensor<TShape, TDType, TDevice>, SyncType>;

  trunc: this['ceil'];

  frac: <TTensor extends Tensor<Shape, GetDType<'F'>>>(
    input: TTensor
  ) => MaybeAsync<TTensor, SyncType>;

  logical_not: <
    TShape extends Shape,
    TDType extends GetDType<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<Tensor<TShape, GetDType<'B'>, TDevice>, SyncType>;

  bitwise_not: <TTensor extends Tensor<Shape, GetDType<'BI'>>>(
    input: TTensor
  ) => MaybeAsync<TTensor, SyncType>;

  isnan: this['logical_not'];

  isinf: <
    TShape extends Shape,
    TDType extends GetDType<'BIFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<Tensor<TShape, GetDType<'B'>, TDevice>, SyncType>;

  isfinite: this['isinf'];

  isreal: this['isinf'];

  isposinf: <
    TShape extends Shape,
    TDType extends GetDType<'BIF'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<Tensor<TShape, GetDType<'B'>, TDevice>, SyncType>;

  isneginf: this['isposinf'];

  angle: <
    TShape extends Shape,
    TDType extends GetDType<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<
    Tensor<
      TShape,
      TDType extends GetDType<'BI'> ? DefaultDType : DTypeRealOf<TDType>,
      TDevice
    >,
    SyncType
  >;

  conj: <TTensor extends Tensor<Shape, GetDType<'BIFCH'>>>(
    input: TTensor
  ) => MaybeAsync<TTensor, SyncType>;

  real: <
    TShape extends Shape,
    TDType extends GetDType<'BIFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<Tensor<TShape, DTypeRealOf<TDType>, TDevice>, SyncType>;

  imag: <
    TShape extends Shape,
    TDType extends GetDType<'CH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<Tensor<TShape, DTypeRealOf<TDType>, TDevice>, SyncType>;

  erf: <
    TShape extends Shape,
    TDType extends GetDType<'BIF'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => MaybeAsync<
    Tensor<
      TShape,
      TDType extends GetDType<'BI'> ? DefaultDType : TDType,
      TDevice
    >,
    SyncType
  >;

  erfc: this['erf'];

  erfinv: this['erf'];

  lgamma: this['erf'];

  digamma: this['erf'];

  sinc: this['sqrt'];

  i0: this['erf'];
}
