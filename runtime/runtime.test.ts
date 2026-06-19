import assert from 'node:assert/strict';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createAPI } from '../index.ts';

const here = dirname(fileURLToPath(import.meta.url));
const api = createAPI({
  defaultDType: 'float32',
  defaultDevice: 'cpu',
  env: {
    PYTHONPATH: join(here, 'test-fixtures')
  }
});

try {
  const left = await api.ones([2, 2]);
  const right = await api.full([2, 2], 2);
  const result = await api.add(left, right);
  const reduced = await api.sum(result, 1);

  assert.deepEqual(left.shape, [2, 2]);
  assert.equal(left.dtype, 'float32');
  assert.equal(left.device, 'cpu');
  assert.match(left.toString(), /^tensor\(/);
  assert.deepEqual(result.shape, [2, 2]);
  assert.deepEqual(reduced.shape, [2]);
  assert.match(String(reduced), /shape=\[2\]/);
  assert.equal(Object.keys(left).includes('handle'), false);
} finally {
  await api.close();
}
