import { Add, And, Bit, Not } from 'ts-arithmetic';
import { Device } from '../basic/device';
import { Tensor } from '../basic/tensor';
import { Equiv, FirstDefined, If } from '../utils/control';
import { NormalizeIndex, RemoveElements, SetElements } from '../utils/tuples';
import { DefaultableDType } from './main';
import { DType } from '../basic/dtype';
import { Shape } from '../basic/shape';
import { DimensionsValidate, DimensionValidate } from './miscValidators';
import { Err, FValidator, IfOk, Ok } from './error';

export interface ReducersAPI<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device,
  SyncType extends 'sync' | 'async'
> {
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
    ...error: ReducerFValidate<S, TDim, TKeepDim>
  ) => Result<S, TInputDType, TInputDevice, TDim, TKeepDim, TDType, SyncType>;

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
    ...error: ReducerFValidate<S, TDim, TKeepDim, 1>
  ) => Result<
    S,
    TInputDType,
    TInputDevice,
    TDim,
    TKeepDim,
    TDType,
    SyncType,
    1
  >;
  nanprod: this['nansum'];
}
export type Result<
  S extends Shape,
  TInputDType extends DType,
  TInputDevice extends Device,
  TDim extends number[] | number | undefined,
  TKeepDim extends boolean | undefined,
  TDType extends DType | undefined,
  SyncType extends 'sync' | 'async',
  AllowKeepDimNoDim extends Bit = 0
> = IfOk<
  ReducerValidate<S, TDim, TKeepDim, AllowKeepDimNoDim>,
  Tensor<
    TDim extends number[]
      ? If<
          FirstDefined<boolean, [TKeepDim], false>,
          SetElements<S, TDim, 1>,
          RemoveElements<S, TDim>
        >
      : TDim extends number
        ? If<
            FirstDefined<boolean, [TKeepDim], false>,
            SetElements<S, [TDim], 1>,
            RemoveElements<S, [TDim]>
          >
        : [],
    FirstDefined<DType, [TDType], TInputDType>,
    TInputDevice
  >,
  SyncType
>;

type ReducerValidate<
  S extends Shape,
  TDim extends number[] | number | undefined = undefined,
  TKeepDim extends boolean | undefined = undefined,
  AllowKeepDimNoDim extends Bit = 0
> = TDim extends number[] | number
  ? DimensionsValidate<S, TDim, 'existing'>
  : If<
      And<Equiv<TKeepDim, true>, Not<AllowKeepDimNoDim>>,
      Err<'cannot set keepdim to true, when dimensions are not specified'>,
      Ok
    >;

type ReducerFValidate<
  S extends Shape,
  TDim extends number[] | number | undefined = undefined,
  TKeepDim extends boolean | undefined = undefined,
  AllowKeepDimNoDim extends Bit = 0
> = FValidator<ReducerValidate<S, TDim, TKeepDim, AllowKeepDimNoDim>>;
