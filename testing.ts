import { API } from './types/api';
import {
  Equiv,
  ReducedValidate,
  ValidateDim,
  ValidateDim_,
  ValidateDimension
} from './types/main';
import { AreAllDTypesPromotable } from './types/basic/dtype';

declare const t: API<'float32', 'cpu'>;

const abc = t.zeros([3, 4], 'float32', 'cpu');

const a = t.cat([abc, abc], 0);

const b = t.stack([abc, abc], -4);

const c = t.isnan(abc);

type A = ValidateDim_<[3, 3], [3], 0>;

type C = ValidateDimension<[3, 3], 3, 0>;

type B = Equiv<[0], []>;

