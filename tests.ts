import { Broadcast } from './types/basic/shape';

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;

type Expect<T extends true> = T;

type BroadcastTests = [
  Expect<Equal<Broadcast<[2, 3], [2, 3]>, [2, 3]>>,
  Expect<Equal<Broadcast<[2, 3], [3]>, [2, 3]>>,
  Expect<Equal<Broadcast<[3], [2, 3]>, [2, 3]>>,

  Expect<Equal<Broadcast<[5, 1, 4], [1, 3, 1]>, [5, 3, 4]>>,
  Expect<Equal<Broadcast<[5, 1, 4], [3, 4]>, [5, 3, 4]>>,
  Expect<Equal<Broadcast<[1, 3, 1], [5, 1, 4]>, [5, 3, 4]>>,

  Expect<Equal<Broadcast<[8, 1, 6, 1], [7, 1, 5]>, [8, 7, 6, 5]>>,
  Expect<Equal<Broadcast<[1, 1, 1], [2, 3, 4]>, [2, 3, 4]>>,
  Expect<Equal<Broadcast<[2, 3, 4], [1]>, [2, 3, 4]>>,

  Expect<Equal<Broadcast<[], [2, 3]>, [2, 3]>>,
  Expect<Equal<Broadcast<[2, 3], []>, [2, 3]>>,
  Expect<Equal<Broadcast<[], []>, []>>,

  Expect<Equal<Broadcast<[1], [7]>, [7]>>,
  Expect<Equal<Broadcast<[7], [1]>, [7]>>,
  Expect<Equal<Broadcast<[1, 1], [9, 8, 7]>, [9, 8, 7]>>,

  Expect<Equal<Broadcast<[2, 3], [4, 3]>, never>>,
  Expect<Equal<Broadcast<[2, 3], [2, 4]>, never>>,
  Expect<Equal<Broadcast<[3], [4]>, never>>,
  Expect<Equal<Broadcast<[2, 1, 3], [4, 2, 3]>, never>>,
  Expect<Equal<Broadcast<[5, 2, 4], [3, 1]>, never>>
];
