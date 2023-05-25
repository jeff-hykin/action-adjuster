# What is this?

The smart way to cache outputs to cold storage.

- auto rebuilds cache when you edit function source code
- uses mutltiprocessing to keep main thread running fast while saving to disk
- excellent change-tracking of arguments thanks to `super_hash`
- can watch change-tracking of external vars and method attributes
- uses python pickle for saving function outputs, if `dill` is available it will use that instead

# How do I use this?

`pip install cool_cache`

```python
from cool_cache import cache, settings

settings.default_folder = None # disable caching to cold-storage, and instead cache to ram
# this is the default, but you can change it
settings.default_folder = "cache.ignore/"

# 
# 
# simple usage (updates whenever function is edited (excluding comments) or when args change)
# 
# 
@cache()
def things_with_args(a,b,c):
    
    from time import sleep; sleep(1) # <- simulating a long-running process
    
    return a + b + c

things_with_args(1,2,3) # not yet cached
things_with_args(1,2,3) # uses cache
things_with_args(9,9,9) # not yet cached
things_with_args(9,9,9) # uses cache


# 
# 
# external vars
# 
# 
external_counter = 0
@cache(depends_on=lambda:[external_counter])
def things_with_external(a,b,c):
    global external_counter
    
    from time import sleep; sleep(1) # <- simulating a long-running process
    
    return external_counter + a + b + c


# 
# behavior
# 
things_with_external(4,5,6) # not yet cached
things_with_external(4,5,6) # uses cache
external_counter = 1
things_with_external(4,5,6) # not yet cached (because external_counter changed)
things_with_external(4,5,6) # uses cache

# 
# 
# filepath arguments
# 
# 
@cache(watch_filepaths=lambda arg1, arg2, arg3: [ arg1, arg2 ]) # because first two args are filepaths
def things_with_files(filepath1, filepath2, c):
    with open(filepath1, 'r') as in_file1:
        with open(filepath2, 'r') as in_file2:
            return in_file1.readlines() + c + in_file2.readlines()

# 
# behavior
# 
things_with_files("./file1.txt", "./file2.txt", "hello")  # not yet cached
things_with_files("./file1.txt", "./file2.txt", "hello")  # cached
with open("./file2.txt",'w') as f: f.write(str(" world")) # <-- modify the file
things_with_files("./file1.txt", "./file2.txt", "hello")  # not yet cached, because file change is detected
things_with_files("./file1.txt", "./file2.txt", "hello")  # cached

# 
# 
# class methods (e.g. self)
# 
# 
class MyThing:
    def __init__(self, path, other_stuff):
        self.path = path
        self.other_stuff = other_stuff
    
    # for example: self.path changing will affect the cache, but self.other_stuff wont affect the cache
    @cache(watch_attributes=lambda self:[ self.path, ])
    def do_some_stuff(self, arg1):
        from time import sleep; sleep(1)
        return self.path + arg1

# 
# bust=True wipes out all cached values for this function on the next run
# 
@cache(bust=True)
def things(a,b,c):
    return 10

```
