import { Add, And } from 'ts-arithmetic';
import { Device } from '../basic/device';
import {
  AreAllDTypesPromotable,
  DType,
  ManyPromotedDType
} from '../basic/dtype';
import { Shape } from '../basic/shape';
import { Tensor } from '../basic/tensor';
import { Equiv, FirstDefined, If } from '../utils/control';
import {
  InsertElements,
  Last,
  NormalizeIndex,
  SetElements
} from '../utils/tuples';
import { Err, FValidator, IfOk, Ok, ValidationLadder } from './error';
import { DefaultableDType } from './main';
import { DimensionFValidate, DimensionValidate } from './miscValidators';

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
