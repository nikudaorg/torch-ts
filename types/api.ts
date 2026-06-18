import { And, Divide, IsInt, IsNotInt, Max, Subtract } from 'ts-arithmetic';
import {
  CatValidate,
  Ceil,
  DefaultableDType,
  Device,
  If,
  Last,
  PickDefined,
  Reduced,
  ReducedValidate,
  Scalar,
  ScalarGetDevice,
  ScalarGetDType,
  Shape,
  Simple,
  SimpleConstructor,
  SimpleValidate,
  StackValidate,
  SumDimNOfTensors,
  Tensor,
  ValidateAllSameDevice,
  ValidateAllSameShapeExceptDim,
  ValidateDim,
  ValidateDimension,
  ValidateManyDTypes,
  ValidateShape
} from './main';
import {
  AllowedCastTargets,
  DType,
  DType_,
  DTypeRealOf,
  Esoteric,
  ManyPromotedDType
} from './type-promotion';
import { InsertElements, RemoveElements, SetElements } from './tuples';

export interface API<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device
> {
  add: <T1 extends Tensor, T2 extends Tensor>(
    input: T1,
    other: T2,
    alpha?: Scalar<T1['device']>,
    ...error: SimpleValidate<T1, T2>
  ) => Simple<T1, T2>;

  sub: this['add'];

  mul: <T1 extends Tensor, T2 extends Tensor>(
    input: T1,
    other: T2,
    ...error: SimpleValidate<T1, T2>
  ) => Simple<T1, T2>;

  //   div:

  zeros: <
    const S extends Shape,
    TDType extends DType | undefined = undefined,
    TDevice extends Device | undefined = undefined
  >(
    shape: S,
    dtype?: TDType,
    device?: TDevice,
    ...error: ValidateShape<S>
  ) => ValidateShape<
    S,
    SimpleConstructor<
      S,
      PickDefined<DType, [TDType], DefaultDType>,
      PickDefined<Device, [TDevice], DefaultDevice>
    >
  >;

  ones: this['zeros'];
  rand: this['zeros']; // for now
  randn: this['zeros'];

  full: <
    const S extends Shape,
    TScalar extends Scalar,
    TDType extends DType | undefined = undefined,
    TDevice extends Device | undefined = undefined
  >(
    shape: S,
    fillValue: TScalar,
    dtype?: TDType,
    device?: TDevice,
    ...error: ValidateShape<S>
  ) => ValidateShape<
    S,
    SimpleConstructor<
      S,
      PickDefined<DType, [TDType, ScalarGetDType<TScalar>], DefaultDType>,
      PickDefined<Device, [TDevice, ScalarGetDevice<TScalar>], DefaultDevice>
    >
  >;

  zeros_like: <
    const S extends Shape,
    TInputDType extends DType,
    TInputDevice extends Device,
    TDType extends DType | undefined = undefined,
    TDevice extends Device | undefined = undefined
  >(
    input: Tensor<S, TInputDType, TInputDevice>,
    dtype?: TDType,
    device?: TDevice
  ) => ValidateShape<
    S,
    SimpleConstructor<
      S,
      PickDefined<DType, [TDType], TInputDType>,
      PickDefined<Device, [TDevice], TInputDevice>
    >
  >;

  ones_like: this['zeros_like'];

  arange: <
    TStart extends number,
    TEnd extends number,
    TStep extends number,
    TDType extends DType | undefined = undefined,
    TDevice extends Device | undefined = undefined
  >(
    start: TStart,
    end: TEnd,
    step?: TStep,
    dtype?: TDType,
    device?: TDevice
  ) => SimpleConstructor<
    [Max<0, Ceil<Divide<Subtract<TEnd, TStart>, TStep>>>], // should probably add some uncertainty here, otherwise size may differ by one from the result.
    PickDefined<
      DType,
      [TDType],
      If<
        And<IsInt<TStart>, And<IsInt<TEnd>, IsInt<TStep>>>,
        'int64',
        DefaultDType
      >
    >,
    PickDefined<Device, [TDevice], DefaultDevice>
  >;

  sum: <
    const S extends Shape,
    TInputDType extends DType,
    TInputDevice extends Device,
    const TDim extends number[] | number | undefined = undefined,
    TKeepDim extends boolean | undefined = undefined,
    TDType extends DType | undefined = undefined
  >(
    input: Tensor<S, TInputDType, TInputDevice>,
    dim?: TDim,
    keepdim?: TKeepDim,
    dtype?: TDType,
    ...error: ReducedValidate<S, TDim, TKeepDim>
  ) => Reduced<S, TInputDType, TInputDevice, TDim, TKeepDim, TDType>;

  mean: this['sum'];
  prod: this['sum'];

  nansum: <
    const S extends Shape,
    TInputDType extends DType,
    TInputDevice extends Device,
    const TDim extends number[] | number | undefined = undefined,
    TKeepDim extends boolean | undefined = undefined,
    TDType extends DType | undefined = undefined
  >(
    input: Tensor<S, TInputDType, TInputDevice>,
    dim?: TDim,
    keepdim?: TKeepDim,
    dtype?: TDType,
    ...error: ReducedValidate<S, TDim, TKeepDim, [], 1>
  ) => Reduced<S, TInputDType, TInputDevice, TDim, TKeepDim, TDType, 1>;
  nanprod: this['nansum'];

  unsqueeze: <
    const S extends Shape,
    TInputDType extends DType,
    TInputDevice extends Device,
    const TDim extends number
  >(
    input: Tensor<S, TInputDType, TInputDevice>,
    dim: TDim,
    ...error: ValidateDimension<S, TDim, 1>
  ) => ValidateDimension<
    S,
    TDim,
    1,
    Tensor<InsertElements<S, [TDim], 1>, TInputDType, TInputDevice>
  >;

  cat: <
    const TTensors extends Tensor[],
    const TDim extends number | undefined = undefined
  >(
    tensors: TTensors,
    dim?: TDim,
    ...error: CatValidate<TTensors, TDim>
  ) => CatValidate<
    TTensors,
    TDim,
    Tensor<
      SetElements<
        Last<TTensors>['shape'],
        [PickDefined<number, [TDim], 0>],
        SumDimNOfTensors<TTensors, PickDefined<number, [TDim], 0>>
      >,
      ManyPromotedDType<TTensors>,
      Last<TTensors>['device']
    >
  >;

  stack: <
    const TTensors extends Tensor[],
    const TDim extends number | undefined = undefined
  >(
    tensors: TTensors,
    dim?: TDim,
    ...error: StackValidate<TTensors, TDim>
  ) => StackValidate<
    TTensors,
    TDim,
    Tensor<
      InsertElements<
        Last<TTensors>['shape'],
        [PickDefined<number, [TDim], 0>],
        TTensors['length']
      >,
      ManyPromotedDType<TTensors>,
      Last<TTensors>['device']
    >
  >;

  abs: <
    TShape extends Shape,
    TDType extends DType_<'IFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<
    TShape,
    TDType extends DType_<'IF'> ? TDType : DTypeRealOf<TDType>,
    TDevice
  >;

  neg: <TTensor extends Tensor<Shape, DType_<'IFCH'>>>(
    input: TTensor
  ) => TTensor;

  sign: <TTensor extends Tensor<Shape, DType_<'BIF'>>>(
    input: TTensor
  ) => TTensor;

  sgn: <TTensor extends Tensor<Shape, DType_<'BIFCH'>>>(
    input: TTensor
  ) => TTensor;

  reciprocal: <
    TShape extends Shape,
    TDType extends DType_<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<
    TShape,
    TDType extends DType_<'BI'> ? DefaultDType : TDType,
    TDevice
  >;

  square: <
    TShape extends Shape,
    TDType extends DType_<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, TDType extends DType_<'B'> ? 'int64' : TDType, TDevice>;

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
    TDType extends DType_<'BIF'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>,
    eps?: number | null
  ) => Tensor<
    TShape,
    TDType extends DType_<'BI'> ? DefaultDType : TDType,
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

  ceil: <TTensor extends Tensor<Shape, DType_<'IF'>>>(
    input: TTensor
  ) => TTensor;

  floor: this['ceil'];

  round: <
    TShape extends Shape,
    TDevice extends Device,
    TDecimals extends number | undefined = undefined,
    TDType extends [TDecimals] extends [undefined]
      ? DType_<'IF'>
      : DType_<'F'> = [TDecimals] extends [undefined]
      ? DType_<'IF'>
      : DType_<'F'>
  >(
    input: Tensor<TShape, TDType, TDevice>,
    decimals?: TDecimals
  ) => Tensor<TShape, TDType, TDevice>;

  trunc: this['ceil'];

  frac: <TTensor extends Tensor<Shape, DType_<'F'>>>(input: TTensor) => TTensor;

  logical_not: <
    TShape extends Shape,
    TDType extends DType_<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, DType_<'B'>, TDevice>;

  bitwise_not: <TTensor extends Tensor<Shape, DType_<'BI'>>>(
    input: TTensor
  ) => TTensor;

  isnan: this['logical_not'];

  isinf: <
    TShape extends Shape,
    TDType extends DType_<'BIFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, DType_<'B'>, TDevice>;

  isfinite: this['isinf'];

  isreal: this['isinf'];

  isposinf: <
    TShape extends Shape,
    TDType extends DType_<'BIF'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, DType_<'B'>, TDevice>;

  isneginf: this['isposinf'];

  angle: <
    TShape extends Shape,
    TDType extends DType_<'BIFC'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<
    TShape,
    TDType extends DType_<'BI'> ? DefaultDType : DTypeRealOf<TDType>,
    TDevice
  >;

  conj: <TTensor extends Tensor<Shape, DType_<'BIFCH'>>>(
    input: TTensor
  ) => TTensor;

  real: <
    TShape extends Shape,
    TDType extends DType_<'BIFCH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, DTypeRealOf<TDType>, TDevice>;

  imag: <
    TShape extends Shape,
    TDType extends DType_<'CH'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<TShape, DTypeRealOf<TDType>, TDevice>;

  erf: <
    TShape extends Shape,
    TDType extends DType_<'BIF'>,
    TDevice extends Device
  >(
    input: Tensor<TShape, TDType, TDevice>
  ) => Tensor<
    TShape,
    TDType extends DType_<'BI'> ? DefaultDType : TDType,
    TDevice
  >;

  erfc: this['erf'];

  erfinv: this['erf'];

  lgamma: this['erf'];

  digamma: this['erf'];

  sinc: this['sqrt'];

  i0: this['erf'];

  //   cos: <TTensor extends Tensor>(input: TTensor) => TTensor;
  //   sin: this['cos'];
  //   sqrt: this['cos'];
  //   neg: this['cos'];
  //   sign: this['cos'];
  //   reciprocal: this['cos'];
  //   square: this['cos'];
  //   rsqrt: this['cos'];
  //   exp: this['cos'];
  //   exp2: this['cos'];
  //   expm1: this['cos'];
  //   log: this['cos'];
  //   log2: this['cos'];
  //   log10: this['cos'];
  //   log1p: this['cos'];
  //   logit: <TTensor extends Tensor>(
  //     input: TTensor,
  //     eps?: Scalar<TTensor['device']>
  //   ) => TTensor;
  //   sigmoid: this['cos'];
  //   tan: this['cos'];
  //   asin: this['cos'];
  //   acos: this['cos'];
  //   atan: this['cos'];
  //   arcsin: this['cos'];
  //   arccos: this['cos'];
  //   arctan: this['cos'];
  //   sinh: this['cos'];
  //   cosh: this['cos'];
  //   tanh: this['cos'];
  //   asinh: this['cos'];
  //   acosh: this['cos'];
  //   atanh: this['cos'];
  //   arcsinh: this['cos'];
  //   arccosh: this['cos'];
  //   arctanh: this['cos'];
  //   ceil: this['cos'];
  //   atanh: this['cos'];
  //   atanh: this['cos'];
  //   log2: this['cos'];
}
