try:
    import requests
except Exception as error:
    requests = None

from .__dependencies__.informative_iterator import ProgressBar
from .__dependencies__.blissful_basics import FS, super_hash
import time 

import os
from os.path import expanduser
home = expanduser('~')

def to_human_time_string(secs):
    secs  = int(round(secs))
    mins  = secs  // 60; secs  = secs  % 60
    hours = mins  // 60; mins  = mins  % 60
    days  = hours // 24; hours = hours % 24
    if days:
        hours = f"{hours}".rjust(2,"0")
        mins  = f"{mins}".rjust(2,"0")
        return f"{days}days, {hours}h:{mins}min"
    elif hours:
        mins  = f"{mins}".rjust(2,"0")
        return f"{hours}h:{mins}min"
    elif mins:
        secs  = f"{secs}".rjust(2,"0")
        return f"{mins}m:{secs}sec"
    else:
        return f"{secs}sec"

class Notifier:
    """
    Example:
        notifier = Notifier(
            disable=False,
            token=secrets.get("telegram_token", None),
        )
        
        # Basic
        notifier.send("Howdy")
        
        # Error handle
        with notifier.when_done:
            raise Exception(f'''Hello Telegram :)''')
        
        # Progress notifications
        # - gives ETA and other time-info
        # - can limit print-rate (by time passed or percent-progress)
        # - will only notify on-print
        for progress, epoch in notifier.progress(range(number_of_epochs), seconds_per_print=100, percent_per_print=2):
            # do stuff
            import random
            accuracy = random.random()
            if accuracy > 0.99:
                notifier.send("Gottem 🎉🎉🎉")
            progress.message = f"accuracy: <code>{accuracy}</code>"
    """
    cache_folder = f"{home}/.cache/telepy/"
    def __init__(
        self,
        *,
        token_path=None,
        message_prefix=None,
        disable=False,
        token_env_var=None,
        token=None,
        chat_id=None,
        parse_mode="HTML",
        bust_cache=False,
    ):
        try:
            self.disable = disable
            self.prefix = f"<b>{message_prefix}</b>: " if message_prefix != None else ""
            self.parse_mode = parse_mode
            self.chat_id = chat_id
            bust_cache and self.bust_cache()
            self.token = token or (token_path and FS.read(token_path)) or (token_env_var and os.environ.get(token_env_var))
            self.token_hash = super_hash(self.token)
            self.chat_id_cache_file = f"{Notifier.cache_folder}/chat_id_{self.token_hash}.txt"
            
            # get chat_id from cache, or from server
            if self.chat_id == None:
                # try to read from cache
                self.chat_id = FS.read(self.chat_id_cache_file)
                
                # only read from server if not disabled
                if not self.chat_id and not self.disable:
                    self.chat_id = self.get_chat_id()
                    # if it worked
                    if self.chat_id != None and not bust_cache and not os.path.exists(self.chat_id_cache_file):
                        # make sure the parent folder exists
                        FS.ensure_is_folder(FS.parent_path(self.chat_id_cache_file), force=False)
                        try:
                            FS.write(data=str(self.chat_id), to=self.chat_id_cache_file, force=False)
                        except Exception as error:
                            pass
            
            if type(self.chat_id) == str:
                self.chat_id = self.chat_id.strip()
            if type(self.token) == str:
                self.token = self.token.strip()
        
            class WhenDone:
                def __init__(this, prefix=None):
                    this.old_prefix = self.prefix
                    this.prefix = str(prefix) if prefix != None else self.prefix
                    self.prefix = this.prefix
                
                def __enter__(this):
                    this.start_time = time.time() 
                    return this
                
                def __exit__(this, _, error, traceback):
                    try:
                        this.end_time = time.time() 
                        duration = to_human_time_string(this.end_time - this.start_time)
                        message = f"process took: {duration}, ended at: {time.ctime()}"
                        if error is not None:
                            message += f" however it ended with an error: {repr(error)}"
                        
                        self.send(message)
                        self.prefix = this.old_prefix
                    except Exception as error:
                        print(f'''Warning: telepy hit an unexpected error in its __exit__ function: ''', error)
                
            
            self.WhenDone = WhenDone
            self.when_done = WhenDone()
        
        except Exception as error:
            print(f'''Warning: telepy hit an unexpected error in its initializer: ''', error)
    
    def bust_cache(self):
        try:
            os.remove(self.chat_id_cache_file)
        except Exception as error:
            pass
    
    def get_chat_id(self):
        try:
            data = { "offset": 0 }
            response = requests.get(f"https://api.telegram.org/bot{self.token}/getUpdates", data=data, timeout=10)
            if response.status_code == 200:
                return response.json()['result'][-1]['message']['chat']['id']
        except Exception as error:
            print("\nHey! telepy couldn't get chat_id! Send your bot a DM, then run this again",)
            return None

    def send(self, message):
        if self.disable:
            return
        try:
            data = {
                "chat_id": self.chat_id,
                "text": self.prefix+str(message)
            }
            if self.parse_mode:
                data["parse_mode"] = self.parse_mode
            response = requests.get(f"https://api.telegram.org/bot{self.token}/sendMessage", data=data, timeout=10)
            if response.status_code != 200 or response.json()["ok"] is not True:
                print(f"Warning: telepy failed to send notification:\n\tstatus_code={response.status_code}\n\tjson:\n\t{response.json()}")
                self.bust_cache() # just to be safe, encase the chat_id from here got corrupted
        except Exception as e:
            print(f"Failed to send notification:\n\texception:\n\t{e}")
    
    def progress(self, *args, **kwargs):
        """
            Example:
                for progress, somthing in notifier.progress([5,6,7,8], seconds_per_print=100, percent_per_print=1):
                    index = progress.index
                    
                    # do stuff
                    import random
                    accuracy = random.random()
                    if accuracy > 0.99:
                        notifier.send("Gottem 🎉🎉🎉")
                    # end do stuff
                    
                    # assign a message
                    progress.message = f"recent accuracy: <code>{accuracy}</code>"
            
            Arguments:
                iterator: any iterable object
                    
                iterations: int
                    if the iterator doesnt have a length
                    then use this keyword arg to tell
                    the progress the expected length
                
                layout: list of str
                    default value: [
                        'title',
                        'bar',
                        'percent',
                        'spacer',
                        'fraction',
                        'spacer',
                        'remaining_time',
                        'spacer',
                        'end_time',
                        'spacer',
                        'duration',
                        'spacer',
                    ]
                
                title: str
                    will be added as a prefix to prints
                
                seconds_per_print: int 
                
                percent_per_print: int
                
                disable_logging: bool
                    for programatic toggling of printing
            
            Summary:
                Progress notifications
                - gives ETA and other time-info
                - can limit print-rate (by time passed or percent-progress)
                - will only notify on-print
            
            Extra Info:
                The progress object (for progress, _ in notifier.progress())
                is a dictionary with the following info:
                    dict(
                        # you can assign values to text and pretex
                        pretext="",         # comes before progress bar
                        text="",            # comes after layout-stuff
                        index=0,            # same as enumerate
                        percent=0,
                        updated=True,       # will only be true after 
                                            # enough time or enough
                                            # %-progress has been made
                        time=self.times[0], # most recent timestamp
                        deviation=None,
                        previous_output="", 
                        expected_number_of_updates_needed=0,
                    )
        """
        def progress_notifier(*args,**kwargs):
            real_title = kwargs.get("title", "")
            title = f"<b>{real_title}</b>\n" if len(real_title) else ""
            for progress, item in ProgressBar(*args,**kwargs):
                if progress.updated: # e.g. update rate can be slower than iteration rate
                    self.send(title+progress.previous_output[len(real_title)+1:].replace("|", "\n|")+progress.get("message", ""))
                yield (progress, item)
            
            self.send(title+progress.previous_output.replace("|", "\n|"))
        
        return progress_notifier(*args, **kwargs)