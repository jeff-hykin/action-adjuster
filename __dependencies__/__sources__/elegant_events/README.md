# What is this?

A simple event manager that works between processes

# How do I use this?

`pip install elegant_events`

file1.py

```python
from elegant_events import Server

# will spinup a server in the background if one isn't already running
server1 = Server("localhost", 7070)

# 
# listener examples:
# 

# attach a callback that triggers every time
@server1.whenever("found_an_event")
def when_event_found(timestamp, data):
    print(f'''data = {data}''')

# attach a callback that will only trigger one time
@server1.once("something_started")
def when_event_found(timestamp, data):
    print(f'''data = {data}''')

# 
# send data example
# 
server1.push(event_name="found_an_event", data="Hello from same process")


# 
# receive/handle events examples
# 

# option 1: manually check (e.g. the cooperative multitasking)
sleep(10) # (run the file2.py while this file is sleeping)
server1.check()
# because `.push()` was called twice, when_event_found() will be triggered twice right here

# option 1: auto-check; (highly responsive and efficient sleep-wait)
# behaves like (but is MUCH better than):
#     while True: server1.check()
server1.listen_forever()
```


file2.py

```python
from elegant_events import Server

# will spinup a server in the background if one isn't already running
server1 = Server("localhost", 7070)

# "data" can be anything that works with json.dumps()
server1.push(event_name="found_an_event", data={"Hello from different process!": True})
```

# API Overview

```py
from elegant_events import Server
server1 = Server("localhost", 7070)

server1.yell(event_name, data)
# only already-started processes will hear this event
# its efficient but only use this if you know the listener will be up-and-running

server1.push(event_name, data, max_size=math.inf)
# similar to yell, BUT processes that start AFTER this function call
# (even 5min after) will *sometimes hear this event
# if max_size=Infinty
#     processes WILL hear every push, no caveats
#     if there are 5 pushes, then 10min later a 
#     process starts and has a whenever("name")
#     that whenever-callback will immediately execute 5 times in a row
# if max_size=1
#     processes will always hear the most-recent push event
#     BUT will not know about any push events before then
# max_size can be any positive integer, as its the size of the buffer
# NOTE: there is only one max_size per event_name, meaning:
#     push("blah_event", max_size=9)
#     push("blah_event", max_size=9)
#     push("blah_event", max_size=1) 
#     other_process.check() # <-- will only see ONE "blah_event" 
#                           # because the push(max_size=1)
#                           # effectively wiped out the buffer

server1.keep_track_of(event_name, should_return_timestamp=False)
# This will start saving events, even there are no callbacks setup
# good to use this immediately after a server is started.
# Once a callback is attached, it will then have the history of the event-occurances

server1.stop_keeping_track_of(event_name, wipeout_backlog=True)
# if wipeout_backlog=True, no callbacks for event_name will trigger on the .check()
# if wipeout_backlog=False, callbacks can still trigger on the next .check()
# so long as the event itself (the .yell() or .push()) occured before the .stop_keeping_track_of()

server1.check(event_name=None, event_names=None)
# asks the server for all the relvent events that occured since the last .check()
# and then triggers the callbacks in chonological order
# by default it gets all events
# event_name="name" will save on bandwidth and only get those events
# event_names=["name1", "name2"] will save on bandwidth and only get those events

server1.locally_trigger(event_name, timestamp, data)
# rarely should this be needed, but if you want to "fake" that an event has occured
# this function will fake it without any interaction with the network
# (e.g. it won't tell any other processes)

server1.whenever(event_name, catch_and_print_errors=True)
# this attaches a callback function
# NOTE: "whenever" really means "whenever", including if the event happened in the past
# catch_and_print_errors=False
#     RARELY will this ever be desireable
#     .check() is when the callbacks are run
#     so, when this is false, if the callback throws an error
#     the .check() will also throw, and will NOT run ANY other
#     callbacks for ANY EVENT (e.g. crashes)
# NOTE: the rare case where .whenever() is called two times on the same function
#       the function will be called twice when the event happen
#       (and will execute n times per event if .whenever() is called times)

server1.once(event_name, catch_and_print_errors=True)
# this also attaches a callback function, which will be detatched after it executes one time
# NOTE: this will execute event if the event happened before the callback was attached
#       which is particularly useful when doing once("something_started") 
#       and the "something_started" event happened before this process started
# for catch_and_print_errors=False, see the .whenever() example above

server1.whenever_future(event_name, catch_and_print_errors=True)
# same as .whenever() but will only trigger for events that happen after this function is defined/attached

server1.once_next(event_name, catch_and_print_errors=True)
# same as .once() but will only trigger for events that happen after this function is defined/attached

server1.remove_callback(event_name, callback)
# NOTE: the rare case where the same callback is attached two times
#       this remove will only remove the first callback
# For more fine-grained control manually edit the server1.callback_entries, ex:
#    for callback_func, catch_and_print_errors, runs_once, time_threshold in server1.callback_entries[event_name]:

server1.listen_forever()
# this established a live connection with the server 
# and any time any event occurs, the callback will immediately run for said event
# (no backlog/batching)
```