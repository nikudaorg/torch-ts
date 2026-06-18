import { And, Eq, GtOrEq, IsInt, Or } from 'ts-arithmetic';
import { Equiv, If } from '../utils/control';
import { Last, Pop } from '../utils/tuples';
import { Err, Ok } from '../api/error';

export type Shape = number[];

export type Broadcast<S1 extends Shape, S2 extends Shape> =
  Equiv<S1, S2> extends 1 ? S1 : Broadcast_<S1, S2>;

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

export type ShapeValidate<S extends Shape> = If<
  IsValidShape<S>,
  Ok,
  Err<`shape must be a list of positive integers`>
>;

type IsValidShape<S extends Shape> = S extends [
  infer First extends number,
  ...infer Rest extends number[]
]
  ? And<IsInt<First>, And<GtOrEq<First, 0>, IsValidShape<Rest>>>
  : 1;
