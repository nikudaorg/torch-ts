import { And, Divide, IsInt, Max, Subtract } from 'ts-arithmetic';
import { DType } from '../basic/dtype';
import { FirstDefined, If } from '../utils/control';
import { DefaultableDType } from './main';
import { Ceil } from '../utils/math';
import { ShapeFValidate } from './miscValidators';
import { IfOk } from './error';
import { Shape, ShapeValidate } from '../basic/shape';
import {
  Scalar,
  ScalarGetDevice,
  ScalarGetDType,
  Tensor
} from '../basic/tensor';
import { Device } from '../basic/device';

export interface ConstructorsAPI<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device
> {
  zeros: <
    const S extends Shape,
    TDType extends DType | undefined = undefined,
    TDevice extends Device | undefined = undefined
  >(
    shape: S,
    dtype?: TDType,
    device?: TDevice,
    ...error: ShapeFValidate<S>
  ) => IfOk<
    ShapeValidate<S>,
    Result<
      S,
      FirstDefined<DType, [TDType], DefaultDType>,
      FirstDefined<Device, [TDevice], DefaultDevice>
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
    ...error: ShapeFValidate<S>
  ) => IfOk<
    ShapeValidate<S>,
    Result<
      S,
      FirstDefined<DType, [TDType, ScalarGetDType<TScalar>], DefaultDType>,
      FirstDefined<Device, [TDevice, ScalarGetDevice<TScalar>], DefaultDevice>
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
  ) => Result<
    S,
    FirstDefined<DType, [TDType], TInputDType>,
    FirstDefined<Device, [TDevice], TInputDevice>
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
  ) => Result<
    [Max<0, Ceil<Divide<Subtract<TEnd, TStart>, TStep>>>], // should probably add some uncertainty here, otherwise size may differ by one from the result.
    FirstDefined<
      DType,
      [TDType],
      If<
        And<IsInt<TStart>, And<IsInt<TEnd>, IsInt<TStep>>>,
        'int64',
        DefaultDType
      >
    >,
    FirstDefined<Device, [TDevice], DefaultDevice>
  >;
}

export type Result<
  S extends Shape,
  TDType extends DType,
  TDevice extends Device
> = Tensor<S, TDType, TDevice>;
