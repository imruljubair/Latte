import test from 'node:test';
import assert from 'node:assert/strict';
import { createStepCache } from '../src/step-cache.mjs';

test('deduplicates requests and evicts least recently used tensor steps', async () => {
  const calls = [];
  const cache = createStepCache(async index => { calls.push(index); return { index }; }, 2);
  await Promise.all([cache.get(0), cache.get(0)]);
  await cache.get(1);
  await cache.get(0);
  await cache.get(2);
  await cache.get(1);
  assert.deepEqual(calls, [0, 1, 2, 1]);
});

test('network errors do not poison future retries', async () => {
  let attempts = 0;
  const cache = createStepCache(async () => {
    if (++attempts === 1) throw new Error('offline');
    return 'recovered';
  });
  await assert.rejects(cache.get(0), /offline/);
  assert.equal(await cache.get(0), 'recovered');
});
