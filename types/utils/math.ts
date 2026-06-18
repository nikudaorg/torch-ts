import { Add, Lt, Subtract } from 'ts-arithmetic';
import { If } from './control';

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
