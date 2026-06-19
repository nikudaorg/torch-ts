import { IsInt } from 'ts-arithmetic';
import { Device } from './device';
import { DType } from './dtype';
import { Shape } from './shape';

export interface Tensor<
  TShape extends Shape = Shape,
  TDType extends DType = DType,
  TDevice extends Device = Device
> {
  readonly kind: 'Tensor';
  readonly shape: TShape;
  readonly dtype: TDType;
  readonly device: TDevice;
  toString(): string;
}

export type Scalar<TDevice extends Device = Device> =
  | Tensor<[], DType, TDevice>
  | number
  | boolean;

export type ScalarGetDType<TScalar extends Scalar> =
  TScalar extends Tensor<infer _, infer TDType, infer _>
    ? TDType
    : Scalar extends boolean
      ? 'bool'
      : TScalar extends number
        ? IsInt<TScalar> extends 1
          ? 'int64'
          : undefined
        : never;

export type ScalarGetDevice<TScalar extends Scalar> =
  TScalar extends Tensor<infer _, infer _, infer TDevice> ? TDevice : undefined;
