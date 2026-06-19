import { createAPI } from './index.ts';

const torch = createAPI({
  defaultDType: 'float16',
  defaultDevice: 'cuda',
  pythonPath: './testing-python/.venv/bin/python'
});

const a = await torch.ones([2, 5]);
const b = await torch.full([2, 3], 4);
const c = await torch.cat([a, b], -1); // hovering in your IDE shows Tensor<[2, 8], 'float16', 'cuda'>
torch.cat([a, b], 0); // all tensors must have the same shape except for the dimension of concatention

await torch.close();
