# torch-ts
__this readme is llm-generated, but the project is (mostly) not__

`torch-ts` is a TypeScript wrapper for PyTorch.

It does not reimplement PyTorch in JavaScript. It starts a Python process, runs PyTorch there, and sends calls to it.

This project is in development. Many PyTorch functions are still missing. The API may change.

## Main point

The main goal is type safety.

Tensor shapes, dtypes, and devices are tracked at the type level. Invalid operations are rejected by TypeScript before runtime.

The type system also computes broadcasting, dtype promotion, and similar rules at the type level. The goal is to stay as close as possible to real PyTorch behavior.

## Example

```ts
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
```

## How it works

1. TypeScript checks tensor rules at compile time.
2. The library starts a Python worker.
3. The worker runs real PyTorch operations.
4. Tensor metadata is returned to TypeScript and kept in the types.

## Setup

You need:

- Node.js
- Python
- PyTorch installed in that Python environment

Install TypeScript dependencies:

```sh
pnpm install
```

Then create or use a Python environment that has `torch` installed, and pass its Python path to `createAPI`.

## Current state

- Runtime calls are async.
- The library currently covers only part of PyTorch.
- Missing functions, edge cases, and behavior mismatches should be expected.
- This is not ready for broad use yet.
