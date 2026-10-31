// Bound client tensor memory; failed requests may be retried.
export function createStepCache(load, limit = 3) {
  const cache = new Map();
  return {
    get(index) {
      if (cache.has(index)) {
        const value = cache.get(index);
        cache.delete(index);
        cache.set(index, value);
        return value;
      }
      const value = Promise.resolve().then(() => load(index)).catch(error => {
        if (cache.get(index) === value) cache.delete(index);
        throw error;
      });
      cache.set(index, value);
      while (cache.size > limit) cache.delete(cache.keys().next().value);
      return value;
    },
  };
}
