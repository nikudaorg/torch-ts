import { Device } from '../basic/device';
import { ConstructorsAPI } from './constructors';
import { ElementWiseBinaryAPI } from './elementwise/binary';
import { ElementWiseUnaryAPI } from './elementwise/unary';
import { ReducersAPI } from './reducers';
import { ReshapeAPI } from './reshape';

export type DefaultableDType = 'float16' | 'bfloat16' | 'float32' | 'float64';

export interface API<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device,
  SyncType extends 'sync' | 'async' = 'async'
>
  extends
    ConstructorsAPI<DefaultDType, DefaultDevice, SyncType>,
    ElementWiseBinaryAPI<DefaultDType, DefaultDevice, SyncType>,
    ElementWiseUnaryAPI<DefaultDType, DefaultDevice, SyncType>,
    ReducersAPI<DefaultDType, DefaultDevice, SyncType>,
    ReshapeAPI<DefaultDType, DefaultDevice, SyncType> {}

export type MaybeAsync<
  T,
  SyncType extends 'sync' | 'async'
> = SyncType extends 'sync' ? T : Promise<T>;
