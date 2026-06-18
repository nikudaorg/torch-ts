import { Device } from '../basic/device';
import { ConstructorsAPI } from './constructors';
import { ElementWiseBinaryAPI } from './elementwise/binary';
import { ElementWiseUnaryAPI } from './elementwise/unary';
import { ReducersAPI } from './reducers';
import { ReshapeAPI } from './reshape';

export type DefaultableDType = 'float16' | 'bfloat16' | 'float32' | 'float64';

export interface API<
  DefaultDType extends DefaultableDType,
  DefaultDevice extends Device
>
  extends
    ConstructorsAPI<DefaultDType, DefaultDevice>,
    ElementWiseBinaryAPI<DefaultDType, DefaultDevice>,
    ElementWiseUnaryAPI<DefaultDType, DefaultDevice>,
    ReducersAPI<DefaultDType, DefaultDevice>,
    ReshapeAPI<DefaultDType, DefaultDevice> {}
