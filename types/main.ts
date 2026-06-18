import {
  Add,
  And,
  Bit,
  Eq,
  Gt,
  GtOrEq,
  IsInt,
  IsNotInt,
  Lt,
  LtOrEq,
  Negate,
  Not,
  Or,
  Subtract
} from 'ts-arithmetic';
import { AreAllDTypesPromotable, DType, PromoteDType } from './type-promotion';
import { HKT, Input, Reduce } from 'yet-another-hkt';
import { NormalizeIndex, RemoveElements, SetElements } from './tuples';

export type Shape = number[];


export type DefaultableDType = 'float16' | 'bfloat16' | 'float32' | 'float64';

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

type GetShape<T extends Tensor> =
  T extends Tensor<infer TShape extends Shape, infer _, infer _>
    ? TShape
    : never;

type IsNever<A> = [A] extends [never] ? 1 : 0;

export type If<A extends Bit | boolean, B, C> = A extends 1 | true
  ? B
  : A extends 0 | false
    ? C
    : B | C;

type Extends<A, B> = A extends B ? 1 : 0;
export type Equiv<A, B> = And<Extends<A, B>, Extends<B, A>>;

type IfExtends<A, B, C, D> = A extends B ? C : D;

type IfEquiv<A, B, C, D> = A extends B ? (B extends A ? C : D) : D;

export type Broadcast<S1 extends Shape, S2 extends Shape> =
  Equiv<S1, S2> extends 1 ? S1 : Broadcast_<S1, S2>;

type Assume<T, U> = T extends U ? T : never;

export type Last<L extends unknown[]> = L extends [...infer _, infer Last]
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

export type SimpleValidate<T1 extends Tensor, T2 extends Tensor, IfOk = []> = If<
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
      IfOk
    >
  >
>;

export type Simple<T1 extends Tensor, T2 extends Tensor> = SimpleValidate<
  T1,
  T2,
  Tensor<
    Assume<Broadcast<T1['shape'], T2['shape']>, Shape>,
    PromoteDType<T1['dtype'], T2['dtype']>,
    T1['device']
  >
>;

export type PickDefined<
  T,
  A extends (T | undefined)[],
  Default extends T
> = Assume<
  A extends [
    infer First extends T | undefined,
    ...infer Rest extends (T | undefined)[]
  ]
    ? Equiv<First, undefined> extends 1
      ? PickDefined<T, Rest, Default>
      : First
    : Default,
  T
>;

// type PickDefined<A extends unknown[], Default> = Assume<
//   PickDefined_<A, Default>
// >;

export type SimpleConstructor<
  S extends Shape,
  TDType extends DType,
  TDevice extends Device
> = Tensor<S, TDType, TDevice>;

export type Floor<N extends number> = `${N}` extends `-0.${string}`
  ? -1
  : `${N}` extends `${infer Before extends number}.${string}`
    ? If<Lt<N, 0>, Subtract<Before, 1>, Before>
    : N;

export type Ceil<N extends number> = `${N}` extends `-0.${string}`
  ? 0
  : `${N}` extends `${infer Before extends number}.${string}`
    ? If<Lt<N, 0>, Before, Add<Before, 1>>
    : N;

export type IsValidShape<S extends Shape> = S extends [
  infer First extends number,
  ...infer Rest extends number[]
]
  ? And<IsInt<First>, And<GtOrEq<First, 0>, IsValidShape<Rest>>>
  : 1;

export type ValidateShape<S extends Shape, IfOk = []> = If<
  IsValidShape<S>,
  IfOk,
  ['Error: shape must be a list of positive integers']
>;

export type ValidateDimension<
  S extends Shape,
  TDim extends number,
  TIncludeNext extends Bit,
  IfOk = []
> = If<
  IsNotInt<TDim>,
  [`Error: non-integer dimension`],
  If<
    TIncludeNext,
    If<
      Or<
        Lt<TDim, Subtract<-1, S['length']>>,
        GtOrEq<TDim, Add<S['length'], 1>>
      >,
      [
        `Error: Dimension out of range (expected to be in range of [${Subtract<-1, S['length']>}, ${S['length']}], but got ${TDim})`
      ],
      IfOk
    >,
    If<
      Or<Lt<TDim, Negate<S['length']>>, GtOrEq<TDim, S['length']>>,
      [
        `Error: Dimension out of range (expected to be in range of [${Negate<S['length']>}, ${Subtract<S['length'], 1>}], but got ${TDim})`
      ],
      IfOk
    >
  >
>;

type HasRepetitions<
  L extends number[],
  Acc extends number = never
> = L extends [infer First extends number, ...infer Rest extends number[]]
  ? First extends Acc
    ? 1
    : HasRepetitions<Rest, Acc | First>
  : 0;

export type ValidateDim_<
  S extends Shape,
  TDim extends number[] | number,
  TIncludeNext extends Bit
> = TDim extends number[]
  ? If<
      HasRepetitions<TDim>,
      ['Error: the dimensions must not repeat'],
      TDim extends [infer First extends number, ...infer Rest extends number[]]
        ? [
            ...ValidateDimension<S, First, TIncludeNext>,
            ...ValidateDim_<S, Rest, TIncludeNext>
          ]
        : []
    >
  : TDim extends number
    ? ValidateDimension<S, TDim, TIncludeNext>
    : never;

export type ValidateDim<
  S extends Shape,
  TDim extends number[] | number,
  TIncludeNext extends Bit,
  IfOk = []
> =
  Equiv<TDim, []> extends 1
    ? ['Error: the list of dimensions must not be empty']
    : ValidateDim_<S, TDim, TIncludeNext> extends []
      ? IfOk
      : ValidateDim_<S, TDim, TIncludeNext>;

export type Reduced<
  S extends Shape,
  TInputDType extends DType,
  TInputDevice extends Device,
  TDim extends number[] | number | undefined = undefined,
  TKeepDim extends boolean | undefined = undefined,
  TDType extends DType | undefined = undefined,
  AllowKeepDimNoDim extends Bit = 0
> = ReducedValidate<
  S,
  TDim,
  TKeepDim,
  Tensor<
    TDim extends number[]
      ? If<
          PickDefined<boolean, [TKeepDim], false>,
          SetElements<S, TDim, 1>,
          RemoveElements<S, TDim>
        >
      : TDim extends number
        ? If<
            PickDefined<boolean, [TKeepDim], false>,
            SetElements<S, [TDim], 1>,
            RemoveElements<S, [TDim]>
          >
        : [],
    PickDefined<DType, [TDType], TInputDType>,
    TInputDevice
  >,
  AllowKeepDimNoDim
>;
type ToBit<A extends boolean> = A extends true
  ? 1
  : A extends false
    ? 0
    : never;

export type ReducedValidate<
  S extends Shape,
  TDim extends number[] | number | undefined = undefined,
  TKeepDim extends boolean | undefined = undefined,
  IfOk = [],
  AllowKeepDimNoDim extends Bit = 0
> = TDim extends number[] | number
  ? ValidateDim<S, TDim, 0, IfOk>
  : If<
      And<Equiv<TKeepDim, true>, Not<AllowKeepDimNoDim>>,
      ['Error: cannot set keepdim to true, when dimensions are not specified.'],
      IfOk
    >;

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

export type AreAllSameKey<
  TTensors extends Tensor[],
  TValue extends Device = TTensors[0]['device']
> = TTensors extends [
  infer First extends Tensor,
  ...infer Rest extends Tensor[]
]
  ? And<Equiv<First['device'], TValue>, AreAllSameKey<Rest, TValue>>
  : 1;

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

export type ValidateAllSameDevice<TTensors extends Tensor[], IfOk = []> = If<
  AreAllSameKey<TTensors>,
  IfOk,
  ['Error: all tensors must be on the same device.']
>;

export type ValidateAllSameShapeExceptDim<
  TTensors extends Tensor[],
  TDim extends number,
  IfOk = []
> = If<
  AreAllSameShapeExceptDim<TTensors, TDim>,
  IfOk,
  [
    'Error: all tensors must have the same shape except for the dimension of concatention'
  ]
>;

export type ValidateAllSameShape<TTensors extends Tensor[], IfOk = []> = If<
  AreAllSameShape<TTensors>,
  IfOk,
  ['Error: all tensors must have the same shape']
>;

export type ValidateManyDTypes<TTensors extends Tensor[], IfOk = []> = If<
  AreAllDTypesPromotable<TTensors>,
  IfOk,
  [
    "Error: couldn't promote the dtypes, cast the tensors to the intended dtypes explicitly"
  ]
>;

type CIfAEquivBElseA<A, B, C> = If<Equiv<A, B>, C, A>;

export type CatValidate<
  TTensors extends Tensor[],
  TDim extends number | undefined,
  IfOk = []
> = CIfAEquivBElseA<
  [
    ...ValidateAllSameDevice<TTensors>,
    ...ValidateManyDTypes<TTensors>,
    ...ValidateAllSameShapeExceptDim<TTensors, PickDefined<number, [TDim], 0>>,
    ...ValidateDimension<
      Last<TTensors>['shape'],
      PickDefined<number, [TDim], 0>,
      0
    >
  ],
  [],
  IfOk
>;

export type StackValidate<
  TTensors extends Tensor[],
  TDim extends number | undefined,
  IfOk = []
> = CIfAEquivBElseA<
  [
    ...ValidateAllSameDevice<TTensors>,
    ...ValidateManyDTypes<TTensors>,
    ...ValidateAllSameShape<TTensors>,
    ...ValidateDimension<
      Last<TTensors>['shape'],
      PickDefined<number, [TDim], 0>,
      1
    >
  ],
  [],
  IfOk
>;
