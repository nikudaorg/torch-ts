import { Not } from 'ts-arithmetic';
import { DType, PromoteDType } from '../../basic/dtype';
import { Assume, Equiv, If, IsNever } from '../../utils/control';
import { Join } from '../../utils/tuples';
import { DefaultableDType } from '../main';
import { Err, FValidator, Ok, IfOk } from '../error';
import { Scalar, Tensor } from '../../basic/tensor';
import { Broadcast, Shape } from '../../basic/shape';
import { Device } from '../../basic/device';

export interface ElementWiseBinaryAPI<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device
> {
  add: <T1 extends Tensor, T2 extends Tensor>(
    input: T1,
    other: T2,
    alpha?: Scalar<T1['device']>,
    ...error: FValidate<T1, T2>
  ) => Result<T1, T2>;

  sub: this['add'];

  mul: <T1 extends Tensor, T2 extends Tensor>(
    input: T1,
    other: T2,
    ...error: FValidate<T1, T2>
  ) => Result<T1, T2>;
}

type Validate<T1 extends Tensor, T2 extends Tensor> = If<
  IsNever<Broadcast<T1['shape'], T2['shape']>>,
  Err<`the size of tensor a (${Join<T1['shape']>}) must be broadcastable to the size of tensor b (${Join<T2['shape']>})`>,
  If<
    IsNever<PromoteDType<T1['dtype'], T2['dtype']>>,
    Err<`couldn't promote the tensor's types ${T1['dtype']} and ${T2['dtype']}`>,
    If<
      Not<Equiv<T1['device'], T2['device']>>,
      Err<`the tensors must share the same device, currently a.device=${T1['device']}, b.device=${T2['device']}`>,
      Ok
    >
  >
>;

type FValidate<T1 extends Tensor, T2 extends Tensor> = FValidator<
  Validate<T1, T2>
>;

type Result<T1 extends Tensor, T2 extends Tensor> = IfOk<
  Validate<T1, T2>,
  Tensor<
    Assume<Broadcast<T1['shape'], T2['shape']>, Shape>,
    PromoteDType<T1['dtype'], T2['dtype']>,
    T1['device']
  >
>;
