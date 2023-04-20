if __name__ == '__main__':
    import sys
    # to be fully pure, the module name is dynamic to allow for any name (two versions of elegant_events could run at the same time under different names)
    exec(f"import {sys.argv[1]} as elegant_events")
    elegant_events.start_server(address=sys.argv[2], port=sys.argv[3])
    exit(0)

from random import random
import json
import time
import traceback
import asyncio
import sys
import math
import urllib.parse

from .__dependencies__ import json_fix
from .__dependencies__ import websockets
from .__dependencies__.websockets.sync.client import connect

connect = websockets.sync.client.connect
this_file = __file__ # seems dumb but will break in interpreter if not assigned to a var
url_encode = lambda string: urllib.parse.quote(string.encode('utf8'))

class Server:
    # TODO: add a get-time endpoint
    # TODO: add a get-id endpoint to allow for shorter ids (to save on bandwith)
    # TODO: add a whenever(only_most_recent) that will look at the backlog-batch and only trigger on the most recent one (intentionally drop packets)
    # TODO: add a "who did what" debugging tool
    # TODO: add a push(to="id") that pre-fills the backlog for a particular client
    def __init__(self, address, port, debugging=False, client_name=None):
        self.address            = address
        self.port               = port
        self.connections        = dict()
        self.client_id          = client_name if client_name != None else f"{random()}"
        self.tracking           = []
        self.url_base           = f"ws://{self.address}:{self.port}"
        self.callback_entries   = {} # key1=event_name, value=list of (callback_func, should_catch_and_print_errors, runs_once, time_threshold)
        self.debugging          = debugging
        self.time_at_prev_check = 0 # start at begining of time, to fetch any/all missed messages
        try:
            debugging and print("trying to connect")
            with connect(f"{self.url_base}/builtin/ping") as websocket:
                debugging and print("sending message")
                websocket.send("ping")
                debugging and print("waiting for message")
                message = websocket.recv()
                if message != "pong":
                    raise Exception(f'''Ping-pong with {self.url_base} failed, received: {repr(message)}''')
        except Exception as error:
            debugging and traceback.print_exc()
            debugging and print(f"starting server, {error}")
            import subprocess
            _process = subprocess.Popen(
                [
                    sys.executable,
                    this_file,
                    __name__,
                    address,
                    f"{port}",
                ],
                **(dict(stdout=sys.stdout) if self.debugging else dict(stdout=subprocess.PIPE)),
                # stderr=subprocess.STDOUT,
            )
            print(f'''server pid: {_process.pid}''')
            # keep trying until it works
            while 1:
                try:
                    with connect(f"{self.url_base}/builtin/ping") as websocket:
                        debugging and print("sending message")
                        websocket.send("ping")
                        debugging and print("waiting for message")
                        message = websocket.recv()
                        if message == "pong":
                            debugging and print("got message")
                            break
                except Exception as error:
                    time.sleep(1)
            
    
    def _builtin_yell(self, path, data):
        path = "builtin/"+path
        if not self.connections.get(path, None):
            self.connections[path] = connect(self.url_base+"/"+url_encode(path))
        self.connections[path].send(json.dumps(data))
        return self.connections[path]
    
    def yell(self, event_name, *, data):
        path = "yell/"+event_name
        if not self.connections.get(path, None):
            self.connections[path] = connect(self.url_base+"/"+url_encode(path))
        self.connections[path].send(json.dumps(data))
        return self.connections[path]
    
    def push(self, event_name, *, data, max_size=math.inf):
        path = "push/"+event_name
        if not self.connections.get(path, None):
            self.connections[path] = connect(self.url_base+"/"+url_encode(path))
        # max_size is a string because json cant handle Infinite
        self.connections[path].send(json.dumps([f"{max_size}", data]))
        return self.connections[path]
    
    def keep_track_of(self, event_name, *, should_return_timestamp=False):
        timestamp = None
        message = [self.client_id, event_name]
        if should_return_timestamp:
            message.append(should_return_timestamp)
        if event_name not in self.tracking:
            timestamp = self._builtin_yell("keep_track_of",[self.client_id, event_name, should_return_timestamp])
            self.tracking.append(event_name)
        # always get the timestamp if one needs to be returned
        if timestamp == None and should_return_timestamp:
            timestamp = self._builtin_yell("keep_track_of",[self.client_id, event_name, should_return_timestamp])
        return timestamp
    
    def stop_keeping_track_of(self, event_name, *, wipeout_backlog=True):
        if event_name in self.tracking:
            self._builtin_yell("stop_keeping_track_of", [self.client_id, event_name, wipeout_backlog])
            if event_name in self.tracking:
                self.tracking.remove(event_name)
    
    def check(self, *, event_name=None, event_names=None):
        if event_name != None:
            event_names = [ event_name ]
        response = self._builtin_yell("check", [self.client_id, event_names ]).recv()
        backlog, timestamp = json.loads(response)
        self.time_at_prev_check = timestamp
        self._process_backlog(backlog)
    
    def _process_backlog(self, backlog):
        chonological_backlog = []
        for each_event_name, events in backlog.items():
             for timestamp, data in events:
                chonological_backlog.append((timestamp, each_event_name, data))
        chonological_backlog.sort()
        for timestamp, each_event_name, data in chonological_backlog:
            data = json.loads(data)
            self.locally_trigger(each_event_name, timestamp=timestamp, data=data)
    
    def locally_trigger(self, event_name, *, timestamp, data):
        new_callbacks_list = []
        for each_callback, catch_and_print_errors, runs_once, time_threshold in self.callback_entries.get(event_name, []):
            if time_threshold <= timestamp:
                if not catch_and_print_errors:
                    each_callback(timestamp, data)
                else:
                    try:
                        each_callback(timestamp, data)
                    except Exception as error:
                        # print the full stack trace, but keep going
                        traceback.print_exc()
                if not runs_once:
                    new_callbacks_list.append((each_callback, catch_and_print_errors, runs_once, time_threshold))
        # new list doesn't have any run_once functions
        self.callback_entries[event_name] = new_callbacks_list
            
    def whenever(self, event_name, *, catch_and_print_errors=True):
        self.keep_track_of(event_name)
        runs_once = False
        time_threshold = 0 # ANYTIME even if the message was sent a year ago, trigger this function as soon as it finds out
        def function_getter(function_being_wrapped):
            self.callback_entries.setdefault(event_name, [])
            self.callback_entries[event_name].append((function_being_wrapped, catch_and_print_errors, runs_once, time_threshold))
            return function_getter
        return function_getter
    
    def once(self, event_name, *, catch_and_print_errors=True):
        self.keep_track_of(event_name)
        runs_once = True
        time_threshold = 0 # ANYTIME even if the message was sent a year ago, trigger this function as soon as it finds out
        def function_getter(function_being_wrapped):
            self.callback_entries.setdefault(event_name, [])
            self.callback_entries[event_name].append((function_being_wrapped, catch_and_print_errors, runs_once, time_threshold))
            return function_getter
        return function_getter
    
    def whenever_future(self, event_name, *, catch_and_print_errors=True):
        time_threshold = self.keep_track_of(event_name, should_return_timestamp=True) # only trigger for events that happened after this tracking was started
        runs_once = False
        def function_getter(function_being_wrapped):
            self.callback_entries.setdefault(event_name, [])
            self.callback_entries[event_name].append((function_being_wrapped, catch_and_print_errors, runs_once, time_threshold))
            return function_getter
        return function_getter
    
    def once_next(self, event_name, *, catch_and_print_errors=True):
        time_threshold = self.keep_track_of(event_name, should_return_timestamp=True)  # only trigger for events that happened after this tracking was started
        def function_getter(function_being_wrapped):
            self.callback_entries.setdefault(event_name, [])
            self.callback_entries[event_name].append((function_being_wrapped, catch_and_print_errors, runs_once, time_threshold))
            return function_getter
        return function_getter
    
    def remove_callback(self, *, event_name, callback):
        self.callback_entries.setdefault(event_name, [])
        self.callback_entries = [
            entry
                for entry in self.callback_entries[event_name]
                    if entry[0] != callback
        ]
        
    def listen_forever(self):
        """
        a much more efficient and responsive alternative to while True: server.check()
        """
        with connect(f"{self.url_base}/builtin/listen") as connection:
            connection.send(json.dumps([self.client_id]))
            while True:
                response = connection.recv()
                backlog = json.loads(response)
                self._process_backlog(backlog)


def start_server(address, port):
    import asyncio
    import argparse 
    import base64
    from collections import defaultdict
    serve = websockets.serve
    
    replay_buffer    = {} # key1=event_name, value=list of (time,message) elements
    active_listeners = {} # key1=process_id, value=websocket for that process
    
    class Client:
        trackers      = defaultdict(lambda *args: []) # key1=event_name, value=list of clients
        clients_by_id = {}
        def __init__(self, id):
            Client.clients_by_id[id] = self
            self.backlog = defaultdict(lambda *args: [])             # key=event_name, value=list of (time,message)
            self.started_listening_to = defaultdict(lambda *args: 0) # key=event_name, value=timestamp
            self.active_listening_socket = None
            
        @staticmethod
        def get_by_id(id):
            if id not in Client.clients_by_id:
                Client.clients_by_id[id] = Client(id)
            return Client.clients_by_id[id]
        
        def track(self, event_name):
            self.backlog.setdefault(event_name, [])
            timestamp = time.time()
            if self not in Client.trackers[event_name]:
                Client.trackers[event_name].append(self)
                self.started_listening_to[event_name] = timestamp
            return timestamp
        
        def stop_tracking(self, event_name, wipeout_backlog):
            if wipeout_backlog and event_name in self.backlog:
                del self.backlog[event_name]
            # keep whats already
            if self in Client.trackers[event_name]:
                Client.trackers[event_name].remove(self)
        
        async def sort_message_into_backlog(self, event_name, timestamp, message):
            self.backlog[event_name].append((timestamp, message))
            # if not using .check()
            if self.active_listening_socket:
                # respond on the other socket
                await self.active_listening_socket.send(json.dumps(self.backlog))
                # clear out the backlog if there was one
                for each_value in self.backlog.values():
                    each_value.clear()
            
            print(f"finished sorting {event_name} into backlog, {self.backlog}")
                
    # 
    # socket setup
    # 
    async def socket_response(websocket):
        async for message in websocket:
            if websocket.path == "/builtin" or websocket.path.startswith("/builtin/"):
                if websocket.path == "/builtin/ping":
                    await websocket.send("pong")
                elif websocket.path == "/builtin/keep_track_of":
                    the_id, event_name, *additional_args = json.loads(message)
                    if len(additional_args) > 0:
                        should_return_timestamp, *additional_args = additional_args
                    client = Client.get_by_id(the_id)
                    timestamp = client.track(event_name)
                    if should_return_timestamp:
                        await websocket.send(timestamp)
                elif websocket.path == "/builtin/stop_keeping_track_of":
                    the_id, event_name, wipeout_backlog  = json.loads(message)
                    client = Client.get_by_id(the_id)
                    client.stop_tracking(event_name, wipeout_backlog)
                elif websocket.path == "/builtin/check":
                    # parse message
                    the_id, the_paths = json.loads(message)
                    
                    # setup structures
                    client = Client.get_by_id(the_id)
                    the_paths = set(client.backlog.keys()) if the_paths == None else set(the_paths)
                    
                    # retrive messages that are older than when the process started listening
                    for each_event_name, packets in replay_buffer.items():
                        if each_event_name in the_paths:
                            for timestamp, message in packets:
                                if timestamp > client.started_listening_to[each_event_name]:
                                    await client.sort_message_into_backlog(each_event_name, timestamp, message)
                    
                    # send the part of the backlog that it asked for
                    return_message = {}
                    for each_event_name in the_paths:
                        return_message[each_event_name] = list(client.backlog.get(each_event_name,[]))
                        client.backlog.get(each_event_name,[]).clear()
                    await websocket.send(json.dumps([return_message, time.time()]))
                elif websocket.path == "/builtin/listen":
                    the_id, *_ = json.loads(message)
                    client = Client.get_by_id(the_id)
                    # retrive messages that are older than when the process started listening
                    for each_event_name, packets in replay_buffer.items():
                        for timestamp, message in packets:
                            if timestamp > client.started_listening_to[each_event_name]:
                                await client.sort_message_into_backlog(each_event_name, timestamp, message)
                    
                    client.active_listening_socket = websocket # even if it already had a socket, only use the latest one
                    
                    # if theres a single message anywhere in the backlog, dump the backlog to the client
                    for each_event_name, packets in client.backlog.items():
                        if len(packets) > 0:
                            await client.active_listening_socket.send(json.dumps(client.backlog))
                            # clear out the backlog if there was one
                            for each_value in client.backlog.values():
                                each_value.clear()
                            break # no need to check other events if one event had a message
                else:
                    print(f"warning, unknown attempt to call a builtin: {websocket.path}")
            elif websocket.path.startswith("/yell/"):
                event_name = urllib.parse.unquote(websocket.path[len("/yell/"):])
                timestamp = time.time()
                for each_client in Client.trackers[event_name]:
                    await each_client.sort_message_into_backlog(event_name, timestamp, message)
            elif websocket.path.startswith("/push/"):
                # parse message
                event_name  = urllib.parse.unquote(websocket.path[len("/push/"):])
                split_index = message.index(",")
                buffer_size = json.loads(message[1:split_index])  # pull out the message size
                message     = message[split_index+1:-1]           # leave the rest of the message in json format
                
                # setup data structures
                replay_buffer.setdefault(event_name, [])
                this_replay_buffer = replay_buffer[event_name]
                
                # actual logic
                timestamp = time.time()
                this_replay_buffer.append((time.time(), message))
                # if finite buffer
                try: buffer_size = int(buffer_size)
                except Exception as error:
                    pass
                if isinstance(buffer_size, int) and buffer_size >= 0:
                    replay_buffer[event_name] = this_replay_buffer[-buffer_size:]
                
                for each_client in Client.trackers[event_name]:
                    await each_client.sort_message_into_backlog(event_name, timestamp, message)
            else:
                print(f"websocket.path: {websocket.path} isn't known")
                
    # 
    # start servers
    # 
    async def main():
        async with serve(socket_response, address, port):
            await asyncio.Future()  # run forever

    asyncio.run(main())
