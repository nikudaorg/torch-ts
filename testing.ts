import { createAPI } from './index.ts';

const torch = createAPI({
  defaultDType: 'float16',
  defaultDevice: 'cpu',
  pythonPath: './testing-python/.venv/bin/python'
});

const a = await torch.ones([2, 5]);
const b = await torch.full([2, 3], 4);
const c = await torch.cat([a, b], -1); // hovering in your IDE shows Tensor<[2, 8], 'float16', 'cuda'>

console.log(c)
// tensor([[1., 1., 1., 1., 1., 4., 4., 4.],
//         [1., 1., 1., 1., 1., 4., 4., 4.]])



// Uncomment for type errors:

// await torch.cat([a, b], 0); // in your editor: **all tensors must have the same shape except for the dimension of concatention**
// const d = await torch.zeros([2, 5], undefined, 'cuda');
// await torch.stack([d, a]); // in your editor: **all tensors must be on the same device**

await torch.close();
