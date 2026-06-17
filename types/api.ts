import { Scalar, SimpleValidate, Tensor } from './main';
import { AllowedCastTargets } from './type-promotion';

type API = {
  add: <T1 extends Tensor, T2 extends Tensor>(
    input: T1,
    other: T2,
    alpha?: Scalar,
    ...rest: SimpleValidate<T1, T2>
  ) => 1;
};
