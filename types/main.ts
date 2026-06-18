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
import { AreAllDTypesPromotable, DType, PromoteDType } from './basic/dtype';
import { HKT, Input, Reduce } from 'yet-another-hkt';
import {
  Join,
  Last,
  NormalizeIndex,
  RemoveElements,
  SetElements
} from './utils/tuples';
import { Assume, Equiv, If, IsNever } from './utils/control';
import { Err, Ok } from './api/error';








