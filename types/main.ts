import { And, Bit, Eq, Not, Or, Subtract } from 'ts-arithmetic';
import { PromoteDType } from './type-promotion';
import { HKT, Input, Reduce } from 'yet-another-hkt';

export type Shape = number[];
export type DType =
  // Boolean
  | 'bool'

  // Unsigned integers
  | 'uint1'
  | 'uint2'
  | 'uint3'
  | 'uint4'
  | 'uint5'
  | 'uint6'
  | 'uint7'
  | 'uint8'
  | 'uint16'
  | 'uint32'
  | 'uint64'

  // Signed integers
  | 'int8'
  | 'int16'
  | 'int32'
  | 'int64'

  // Floating point
  | 'float4_e2m1fn_x2'
  | 'float8_e4m3fn'
  | 'float8_e4m3fnuz'
  | 'float8_e5m2'
  | 'float8_e5m2fnuz'
  | 'float8_e8m0fnu'
  | 'float16'
  | 'bfloat16'
  | 'float32'
  | 'float64'

  // Complex
  | 'complex32'
  | 'complex64'
  | 'complex128'

  // Quantized
  | 'qint8'
  | 'quint8'
  | 'qint32'
  | 'quint4x2'
  | 'quint2x4';

export type Device = string;

export interface Tensor<
  TShape extends Shape = Shape,
  TDType extends DType = DType,
  TDevice extends Device = Device
> {
  kind: 'Tensor';
  shape: TShape;
  dtype: TDType;
  device: TDevice;
}

export type Scalar<
  TDType extends DType = DType,
  TDevice extends Device = Device
> = Tensor<[], TDType, TDevice>;

type GetShape<T extends Tensor> =
  T extends Tensor<infer TShape extends Shape, infer _, infer _>
    ? TShape
    : never;

type IsNever<A> = [A] extends [never] ? 1 : 0;

type If<A extends Bit, B, C> = A extends 1 ? B : A extends 0 ? C : B | C;

type Extends<A, B> = A extends B ? 1 : 0;
type Equiv<A, B> = And<Extends<A, B>, Extends<B, A>>;

type IfExtends<A, B, C, D> = A extends B ? C : D;

type IfEquiv<A, B, C, D> = A extends B ? (B extends A ? C : D) : D;

export type Broadcast<S1 extends Shape, S2 extends Shape> =
  Equiv<S1, S2> extends 1 ? S1 : Broadcast_<S1, S2>;

type Last<L extends unknown[]> = L extends [...infer _, infer Last]
  ? Last
  : never;
type Pop<L extends unknown[]> = L extends [...infer Rest, infer _]
  ? Rest
  : never;

type Broadcast_<S1 extends Shape, S2 extends Shape> =
  Equiv<S1, []> extends 1
    ? S2
    : Equiv<S2, []> extends 1
      ? S1
      : Or<
            Eq<Last<S1>, Last<S2>>,
            Or<Eq<Last<S1>, 1>, Eq<Last<S2>, 1>>
          > extends 1
        ? [
            ...Broadcast_<Pop<S1>, Pop<S2>>,
            Last<Eq<Last<S1>, 1> extends 1 ? S2 : S1>
          ]
        : never;

// export type SimpleValidate<T1 extends Tensor, T2 extends Tensor> = If<
//   Or<
//     IsNever<Broadcast<T1['shape'], T2['shape']>>,
//     Or<
//       IsNever<PromoteDType<T1['dtype'], T2['dtype']>>,
//       Not<Equiv<T1['device'], T2['device']>>
//     >
//   >,
//   2
// >;

interface ConcatNumbersHKT<TSeparator extends string> extends HKT<{
  acc: string;
  element: number;
}> {
  output: Input<this>['acc'] extends ''
    ? Input<this>['element']
    : `${Input<this>['acc']}${TSeparator}${Input<this>['element']}`;
  // not the best implementation, reduce should allow taking the first element as the initial (+ a default in that case)
}

type RenderShape<S extends Shape> = `(${Reduce<
  string,
  ConcatNumbersHKT<', '>,
  '',
  S
>})`;

export type SimpleValidate<T1 extends Tensor, T2 extends Tensor> = If<
  IsNever<Broadcast<T1['shape'], T2['shape']>>,
  [
    `Error: the size of tensor a ${RenderShape<T1['shape']>} must be broadcastable to the size of tensor b ${RenderShape<T2['shape']>}`
  ],
  If<
    IsNever<PromoteDType<T1['dtype'], T2['dtype']>>,
    [
      `Error: couldn't promote the tensor's types ${T1['dtype']} and ${T2['dtype']}`
    ],
    If<
      Not<Equiv<T1['device'], T2['device']>>,
      [
        `Error: the tensors must share the same device, currently a.device=${T1['device']}, b.device=${T2['device']}`
      ],
      []
    >
  >
>;

export type Simple<
  T1 extends Tensor,
  T2 extends Tensor,
  S1 extends Shape,
  S2 extends Shape
> = Tensor<Broadcast<S1, S2>, T1['dtype']>;
