from super_hash import super_hash

def cache_outputs(function):
    cache = {}
    def caching_function(*args, **kwargs):
        key = super_hash((args, kwargs))
        return cache.get(
            key,
            cache.setdefault(key, function(*args, **kwargs))
        )
    return caching_function