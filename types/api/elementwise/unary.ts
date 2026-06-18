import { Device } from '../../basic/device';
import { DTypeRealOf, GetDType } from '../../basic/dtype';
import { Shape } from '../../basic/shape';
import { Tensor } from '../../basic/tensor';
import { DefaultableDType } from '../main';

export interface ElementWiseUnaryAPI<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device
> {
  abs: <
    TShape extends Shape,
    TDType extends GetDType<'IFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<
    TShape,
    TDType extends GetDType<'IF'> ? TDType : DTypeRealOf<TDType>,
    TDevice
  >;

  neg: <TTensor extends Tensor<Shape, GetDType<'IFCH'>>>(
    input: TTensor
  ) => TTensor;

  sign: <TTensor extends Tensor<Shape, GetDType<'BIF'>>>(
    input: TTensor
  ) => TTensor;

  sgn: <TTensor extends Tensor<Shape, GetDType<'BIFCH'>>>(
    input: TTensor
  ) => TTensor;

  reciprocal: <
    TShape extends Shape,
    TDType extends GetDType<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<
    TShape,
    TDType extends GetDType<'BI'> ? DefaultDType : TDType,
    TDevice
  >;

  square: <
    TShape extends Shape,
    TDType extends GetDType<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, TDType extends GetDType<'B'> ? 'int64' : TDType, TDevice>;

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
  ) => Tensor<
    TShape,
    TDType extends GetDType<'BI'> ? DefaultDType : TDType,
    TDevice
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
  ) => TTensor;

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
  ) => Tensor<TShape, TDType, TDevice>;

  trunc: this['ceil'];

  frac: <TTensor extends Tensor<Shape, GetDType<'F'>>>(
    input: TTensor
  ) => TTensor;

  logical_not: <
    TShape extends Shape,
    TDType extends GetDType<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, GetDType<'B'>, TDevice>;

  bitwise_not: <TTensor extends Tensor<Shape, GetDType<'BI'>>>(
    input: TTensor
  ) => TTensor;

  isnan: this['logical_not'];

  isinf: <
    TShape extends Shape,
    TDType extends GetDType<'BIFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, GetDType<'B'>, TDevice>;

  isfinite: this['isinf'];

  isreal: this['isinf'];

  isposinf: <
    TShape extends Shape,
    TDType extends GetDType<'BIF'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, GetDType<'B'>, TDevice>;

  isneginf: this['isposinf'];

  angle: <
    TShape extends Shape,
    TDType extends GetDType<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<
    TShape,
    TDType extends GetDType<'BI'> ? DefaultDType : DTypeRealOf<TDType>,
    TDevice
  >;

  conj: <TTensor extends Tensor<Shape, GetDType<'BIFCH'>>>(
    input: TTensor
  ) => TTensor;

  real: <
    TShape extends Shape,
    TDType extends GetDType<'BIFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, DTypeRealOf<TDType>, TDevice>;

  imag: <
    TShape extends Shape,
    TDType extends GetDType<'CH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, DTypeRealOf<TDType>, TDevice>;

  erf: <
    TShape extends Shape,
    TDType extends GetDType<'BIF'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<
    TShape,
    TDType extends GetDType<'BI'> ? DefaultDType : TDType,
    TDevice
  >;

  erfc: this['erf'];

  erfinv: this['erf'];

  lgamma: this['erf'];

  digamma: this['erf'];

  sinc: this['sqrt'];

  i0: this['erf'];
}
