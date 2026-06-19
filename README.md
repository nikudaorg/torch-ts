# torch-ts

An asynchronous TypeScript API backed by a dedicated Python/PyTorch process.
Tensor storage never crosses the process boundary: TypeScript tensors contain
runtime metadata and an internal opaque handle only.

```ts
import { createAPI } from './index.ts';

const torch = createAPI({
  defaultDType: 'float32',
  defaultDevice: 'cpu'
});

const a = await torch.ones([2, 3]);
const b = await torch.square(a);

console.log(b.toString()); // synchronous; uses the repr cached with the result
await torch.close();
```

`python3` must resolve to an environment containing PyTorch. `pythonPath` can
select a virtual environment explicitly.

## Process protocol

Node starts one Python worker per API instance. Messages use the same framing
and JSON representation on all platforms:

1. 4-byte unsigned big-endian payload length.
2. UTF-8 JSON request or response.
3. Tensor arguments are encoded as `{ "$tensor": handle }`.
4. Tensor results include `handle`, `shape`, `dtype`, `device`, and `repr`.

The byte stream is a Unix domain socket on Unix-like systems and a named pipe
on Windows. This platform-specific part is isolated in the transport edge.
