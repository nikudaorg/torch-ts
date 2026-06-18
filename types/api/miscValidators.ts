import {
  Add,
  Bit,
  GtOrEq,
  IsNotInt,
  Lt,
  Negate,
  Or,
  Subtract
} from 'ts-arithmetic';
import { Equiv, If } from '../utils/control';
import { Err, FValidator, Ok, IfOk } from './error';
import { HasDuplicates } from '../utils/tuples';
import { Shape, ShapeValidate } from '../basic/shape';

export type ShapeFValidate<S extends Shape> = FValidator<ShapeValidate<S>>;

export type DimensionValidate<
  S extends Shape,
  TDim extends number,
  TType extends 'existing' | 'new'
> = If<
  IsNotInt<TDim>,
  Err<`non-integer dimension`>,
  If<
    Equiv<TType, 'new'>,
    If<
      Or<
        Lt<TDim, Subtract<-1, S['length']>>,
        GtOrEq<TDim, Add<S['length'], 1>>
      >,
      Err<`dimension out of range (expected to be in range of [${Subtract<-1, S['length']>}, ${S['length']}], but got ${TDim})`>,
      Ok
    >,
    If<
      Or<Lt<TDim, Subtract<0, S['length']>>, GtOrEq<TDim, S['length']>>,
      Err<`dimension out of range (expected to be in range of [${Negate<S['length']>}, ${Subtract<S['length'], 1>}], but got ${TDim})`>,
      Ok
    >
  >
>;

export type DimensionFValidate<
  S extends Shape,
  TDim extends number,
  TType extends 'existing' | 'new'
> = FValidator<DimensionValidate<S, TDim, TType>>;

export type DimensionsValidateInner<
  S extends Shape,
  TDim extends number[] | number,
  TType extends 'existing' | 'new'
> = TDim extends number[]
  ? If<
      HasDuplicates<TDim>,
      Err<'the dimensions must not repeat'>,
      TDim extends [infer First extends number, ...infer Rest extends number[]]
        ? IfOk<
            DimensionValidate<S, First, TType>,
            DimensionsValidateInner<S, Rest, TType>
          >
        : Ok
    >
  : TDim extends number
    ? DimensionValidate<S, TDim, TType>
    : never;

export type DimensionsValidate<
  S extends Shape,
  TDim extends number[] | number,
  TType extends 'existing' | 'new'
> =
  Equiv<TDim, []> extends 1
    ? Err<'Error: the list of dimensions must not be empty'>
    : DimensionsValidateInner<S, TDim, TType> extends []
      ? Ok
      : DimensionsValidateInner<S, TDim, TType>;

export type DimensionsFValidate<
  S extends Shape,
  TDim extends number[] | number,
  TType extends 'existing' | 'new'
> = FValidator<DimensionsValidate<S, TDim, TType>>;