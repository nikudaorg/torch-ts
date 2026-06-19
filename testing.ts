import { createAPI } from '.';

const torch = createAPI({
  defaultDType: 'float32',
  defaultDevice: 'cpu',
  pythonPath: './testing-python/.venv/bin/python'
});

const tensor1 = await torch.ones([2, 3]);
const tensor2 = await torch.full([], 5);
const tensor3 = await torch.add(tensor1, tensor2);

console.log(tensor1.toString());
console.log(tensor2.toString());
console.log(tensor3.toString());

await torch.close();
