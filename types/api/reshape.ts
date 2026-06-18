import { Add, And, Lt, Multiply } from 'ts-arithmetic';
import { Device } from '../basic/device';
import {
  AreAllDTypesPromotable,
  DType,
  ManyPromotedDType
} from '../basic/dtype';
import { Shape, ShapeValidate } from '../basic/shape';
import { Tensor } from '../basic/tensor';
import { Equiv, FirstDefined, If, IsNever } from '../utils/control';
import {
  InsertElements,
  Last,
  NormalizeIndex,
  RemoveElements,
  SetElements,
  SwapElements
} from '../utils/tuples';
import { Err, FValidator, IfOk, Ok, ValidationLadder } from './error';
import { DefaultableDType } from './main';
import {
  DimensionFValidate,
  DimensionsFValidate,
  DimensionsValidate,
  DimensionValidate
} from './miscValidators';

export interface ReshapeAPI<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device
> {
  unsqueeze: <
    const S extends Shape,
    TInputDType extends DType,
    TInputDevice extends Device,
    const TDim extends number
  >(
    input: Tensor<S, TInputDType, TInputDevice>,
    dim: TDim,
    ...error: DimensionFValidate<S, TDim, 'new'>
  ) => IfOk<
    DimensionValidate<S, TDim, 'new'>,
    Tensor<InsertElements<S, [TDim], 1>, TInputDType, TInputDevice>
  >;

  squeeze: <
    const S extends Shape,
    TInputDType extends DType,
    TInputDevice extends Device,
    const TDim extends number[] | number | undefined = undefined
  >(
    input: Tensor<S, TInputDType, TInputDevice>,
    dim?: TDim,
    ...error: FValidator<SqueezeValidate<S, TDim>>
  ) => IfOk<
    SqueezeValidate<S, TDim>,
    TDim extends number[] | number
      ? Tensor<SqueezeAt<S, TDim>, TInputDType, TInputDevice>
      : Tensor<SqueezeAll<S>, TInputDType, TInputDevice>
  >;

  transpose: <
    const S extends Shape,
    TInputDType extends DType,
    TInputDevice extends Device,
    const TDim0 extends number,
    const TDim1 extends number
  >(
    input: Tensor<S, TInputDType, TInputDevice>,
    dim0: TDim0,
    dim1: TDim1,
    ...error: FValidator<TransposeValidate<S, TDim0, TDim1>>
  ) => IfOk<
    TransposeValidate<S, TDim0, TDim1>,
    Tensor<SwapElements<S, TDim0, TDim1>, TInputDType, TInputDevice>
  >;

  //   repeat: <
  //     const S extends Shape,
  //     TInputDType extends DType,
  //     TInputDevice extends Device,
  //     const TRepeats extends Shape
  //   >(
  //     input: Tensor<S, TInputDType, TInputDevice>,
  //     repeats: TRepeats,
  //     ...error: FValidator<RepeatValidate<S, TRepeats>>
  //   ) => IfOk<
  //     RepeatValidate<S, TRepeats>,
  //     Tensor<RepeatShape<S, TRepeats>, TInputDType, TInputDevice>
  //   >;

  cat: <
    const TTensors extends Tensor[],
    const TDim extends number | undefined = undefined
  >(
    tensors: TTensors,
    dim?: TDim,
    ...error: FValidator<CatValidate<TTensors, TDim>>
  ) => IfOk<
    CatValidate<TTensors, TDim>,
    Tensor<
      SetElements<
        Last<TTensors>['shape'],
        [FirstDefined<number, [TDim], 0>],
        SumDimNOfTensors<TTensors, FirstDefined<number, [TDim], 0>>
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
    ...error: FValidator<StackValidate<TTensors, TDim>>
  ) => IfOk<
    StackValidate<TTensors, TDim>,
    Tensor<
      InsertElements<
        Last<TTensors>['shape'],
        [FirstDefined<number, [TDim], 0>],
        TTensors['length']
      >,
      ManyPromotedDType<TTensors>,
      Last<TTensors>['device']
    >
  >;
}

type AreAllSameDevice<
  TTensors extends Tensor[],
  TValue extends Device = TTensors[0]['device']
> = TTensors extends [
  infer First extends Tensor,
  ...infer Rest extends Tensor[]
]
  ? And<Equiv<First['device'], TValue>, AreAllSameDevice<Rest, TValue>>
  : 1;

export type SumDimNOfTensors<
  TTensors extends Tensor[],
  TDim extends number
> = TTensors extends [
  infer First extends Tensor,
  ...infer Rest extends Tensor[]
]
  ? Add<
      First['shape'][NormalizeIndex<TDim, First['shape']['length']>],
      SumDimNOfTensors<Rest, TDim>
    >
  : 0;

export type AreAllSameShapeExceptDim<
  TTensors extends Tensor[],
  TDim extends number,
  TValue extends Shape = SetElements<TTensors[0]['shape'], [TDim], 'skip'>
> = TTensors extends [
  infer First extends Tensor,
  ...infer Rest extends Tensor[]
]
  ? And<
      Equiv<SetElements<First['shape'], [TDim], 'skip'>, TValue>,
      AreAllSameShapeExceptDim<Rest, TDim, TValue>
    >
  : 1;

export type AreAllSameShape<
  TTensors extends Tensor[],
  TValue extends Shape = TTensors[0]['shape']
> = TTensors extends [
  infer First extends Tensor,
  ...infer Rest extends Tensor[]
]
  ? And<Equiv<First['shape'], TValue>, AreAllSameShape<Rest, TValue>>
  : 1;

export type ValidateAllSameDevice<TTensors extends Tensor[]> = If<
  AreAllSameDevice<TTensors>,
  Ok,
  Err<'all tensors must be on the same device'>
>;

export type ValidateAllSameShapeExceptDim<
  TTensors extends Tensor[],
  TDim extends number
> = If<
  AreAllSameShapeExceptDim<TTensors, TDim>,
  Ok,
  Err<'all tensors must have the same shape except for the dimension of concatention'>
>;

export type ValidateAllSameShape<TTensors extends Tensor[]> = If<
  AreAllSameShape<TTensors>,
  Ok,
  Err<'Error: all tensors must have the same shape'>
>;

export type ValidateManyDTypes<TTensors extends Tensor[]> = If<
  AreAllDTypesPromotable<TTensors>,
  Ok,
  Err<"couldn't promote the dtypes, cast the tensors to the intended dtypes explicitly">
>;

type CIfAEquivBElseA<A, B, C> = If<Equiv<A, B>, C, A>;

export type SqueezeAll<S extends Shape> = S extends [
  infer First extends number,
  ...infer Rest extends Shape
]
  ? Equiv<First, 1> extends 1
    ? SqueezeAll<Rest>
    : [First, ...SqueezeAll<Rest>]
  : [];

type SqueezeValidate<
  S extends Shape,
  TDim extends number | number[] | undefined
> = TDim extends number | number[]
  ? IfOk<
      DimensionsValidate<S, TDim, 'existing'>,
      If<
        IsNever<SqueezeAt<S, TDim>>,
        Err<'when specifying dimensions for squeeze, all specified dimensions must be of size 1'>,
        Ok
      >
    >
  : Ok;

type SqueezeAt<
  S extends Shape,
  TDim extends number[] | number
> = TDim extends number
  ? SqueezeAt<S, [TDim]>
  : TDim extends number[]
    ? If<
        Equiv<S[NormalizeIndex<TDim[number], S['length']>], 1>,
        RemoveElements<S, TDim>,
        never
      >
    : never;

// type PadShapeLeftImpl<S extends Shape, TTargetLength extends number> =
//   Lt<S['length'], TTargetLength> extends 1
//     ? PadShapeLeftImpl<[1, ...S], TTargetLength>
//     : S;

// type PadShapeLeft<
//   S extends Shape,
//   TTargetLength extends number
// > = PadShapeLeftImpl<S, TTargetLength>;

// type MultiplyShapes<S1 extends Shape, S2 extends Shape> = S1 extends [
//   infer First1 extends number,
//   ...infer Rest1 extends Shape
// ]
//   ? S2 extends [infer First2 extends number, ...infer Rest2 extends Shape]
//     ? [Multiply<First1, First2>, ...MultiplyShapes<Rest1, Rest2>]
//     : []
//   : [];

// export type RepeatShape<
//   S extends Shape,
//   TRepeats extends Shape
// > = MultiplyShapes<PadShapeLeft<S, TRepeats['length']>, TRepeats>;

// type ValidateRepeatRank<S extends Shape, TRepeats extends Shape> = If<
//   Lt<TRepeats['length'], S['length']>,
//   Err<'the number of repeat dimensions must not be smaller than the number of dimensions of the tensor'>,
//   Ok
// >;

// export type RepeatValidate<
//   S extends Shape,
//   TRepeats extends Shape
// > = ValidationLadder<
//   [ShapeValidate<TRepeats>, ValidateRepeatRank<S, TRepeats>]
// >;

export type TransposeValidate<
  S extends Shape,
  TDim0 extends number,
  TDim1 extends number
> = ValidationLadder<
  [
    DimensionValidate<S, TDim0, 'existing'>,
    DimensionValidate<S, TDim1, 'existing'>
  ]
>;

export type CatValidate<
  TTensors extends Tensor[],
  TDim extends number | undefined
> = ValidationLadder<
  [
    ValidateAllSameDevice<TTensors>,
    ValidateManyDTypes<TTensors>,
    ValidateAllSameShapeExceptDim<TTensors, FirstDefined<number, [TDim], 0>>,
    DimensionValidate<
      Last<TTensors>['shape'],
      FirstDefined<number, [TDim], 0>,
      'existing'
    >
  ]
>;

export type StackValidate<
  TTensors extends Tensor[],
  TDim extends number | undefined
> = ValidationLadder<
  [
    ValidateAllSameDevice<TTensors>,
    ValidateManyDTypes<TTensors>,
    ValidateAllSameShape<TTensors>,
    DimensionValidate<
      Last<TTensors>['shape'],
      FirstDefined<number, [TDim], 0>,
      'new'
    >
  ]
>;
