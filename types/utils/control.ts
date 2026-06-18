import { And, Bit } from 'ts-arithmetic';

export type Extends<A, B> = A extends B ? 1 : 0;
export type Equiv<A, B> = And<Extends<A, B>, Extends<B, A>>;

export type IsNever<A> = [A] extends [never] ? 1 : 0;

export type If<A extends Bit | boolean, B, C> = A extends 1 | true
  ? B
  : A extends 0 | false
    ? C
    : B | C;

export type Assume<T, U> = T extends U ? T : never;

export type FirstDefined<
  T,
  A extends (T | undefined)[],
  Default extends T
> = Assume<
  A extends [
    infer First extends T | undefined,
    ...infer Rest extends (T | undefined)[]
  ]
    ? Equiv<First, undefined> extends 1
      ? FirstDefined<T, Rest, Default>
      : First
    : Default,
  T
>;

export type ToBit<A extends boolean> = A extends true
  ? 1
  : A extends false
    ? 0
    : never;
