import type { Add, Lt } from 'ts-arithmetic';
import { If } from './control';

export type NormalizeIndex<
  Index extends number,
  Length extends number
> = Index extends Index ? If<Lt<Index, 0>, Add<Length, Index>, Index> : never;

type RemoveElementsImpl<
  L extends unknown[],
  RemovedIndices extends number,
  Position extends unknown[] = [],
  Result extends unknown[] = []
> = L extends [infer Head, ...infer Tail]
  ? Position['length'] extends RemovedIndices
    ? RemoveElementsImpl<Tail, RemovedIndices, [...Position, unknown], Result>
    : RemoveElementsImpl<
        Tail,
        RemovedIndices,
        [...Position, unknown],
        [...Result, Head]
      >
  : Result;

export type RemoveElements<L extends unknown[], I extends number[]> =
  // Ordinary arrays do not have statically known positions.
  number extends L['length']
    ? L
    : RemoveElementsImpl<L, NormalizeIndex<I[number], L['length']>>;

type NormalizeInsertionIndex<
  Index extends number,
  Length extends number
> = Index extends Index
  ? `${Index}` extends `-${string}`
    ? // For insertion positions, -1 means the position after the last element.
      Add<Length, Add<Index, 1>>
    : Index
  : never;

type InsertionsAt<
  Indices extends number[],
  Position extends number,
  Length extends number,
  Value,
  Result extends unknown[] = []
> = Indices extends [infer Index extends number, ...infer Rest extends number[]]
  ? NormalizeInsertionIndex<Index, Length> extends Position
    ? InsertionsAt<Rest, Position, Length, Value, [...Result, Value]>
    : InsertionsAt<Rest, Position, Length, Value, Result>
  : Result;

type InsertElementsImpl<
  Remaining extends unknown[],
  Indices extends number[],
  Value,
  OriginalLength extends number,
  Position extends unknown[] = [],
  Result extends unknown[] = []
> = Remaining extends [infer Head, ...infer Tail]
  ? InsertElementsImpl<
      Tail,
      Indices,
      Value,
      OriginalLength,
      [...Position, unknown],
      [
        ...Result,
        ...InsertionsAt<Indices, Position['length'], OriginalLength, Value>,
        Head
      ]
    >
  : [
      ...Result,
      ...InsertionsAt<Indices, Position['length'], OriginalLength, Value>
    ];

export type InsertElements<
  L extends unknown[],
  I extends number[],
  Value
> = number extends L['length']
  ? L
  : InsertElementsImpl<L, I, Value, L['length']>;

export type SetElements<L extends unknown[], I extends number[], A> = {
  [K in keyof L]: K extends `${infer N extends number}`
    ? N extends NormalizeIndex<I[number], L['length']>
      ? A
      : L[K]
    : L[K];
};

export type Last<L extends unknown[]> = L extends [...infer _, infer Last]
  ? Last
  : never;
export type Pop<L extends unknown[]> = L extends [...infer Rest, infer _]
  ? Rest
  : never;

type Printable = string | number | bigint | boolean | null | undefined;

export type Join<L extends Printable[], Sep extends string = ', '> = L extends [
  infer First extends Printable,
  infer Second extends Printable,
  ...infer Rest extends Printable[]
]
  ? `${First}${Sep}${Join<[Second, ...Rest], Sep>}`
  : L extends [infer First extends Printable]
    ? `${First}`
    : '';

export type HasDuplicates<
  L extends number[],
  Acc extends number = never
> = L extends [infer First extends number, ...infer Rest extends number[]]
  ? First extends Acc
    ? 1
    : HasDuplicates<Rest, Acc | First>
  : 0;
