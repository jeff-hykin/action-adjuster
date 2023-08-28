try:
    import requests
except Exception as error:
    pass

class TelegramNotifier:
    def __init__(self, token: str, parse_mode: str = None, chat_id: str = None):
        self._token = token
        self._parse_mode = parse_mode
        if chat_id is None:
            self._get_chat_id()
        else:
            self._chat_id = chat_id

    def _get_chat_id(self):
        try:
            data = {
                "offset": 0
            }
            response = requests.get(f"https://api.telegram.org/bot{self._token}/getUpdates", data=data, timeout=10)
            if response.status_code == 200:
                self._chat_id = response.json()['result'][-1]['message']['chat']['id']
        except Exception as e:
            self._chat_id = None
            print("Couldn't get chat_id! Try sending a message to the bot first\n\t", e)

    def send(self, msg: str):
        if self._chat_id is None:
            self._get_chat_id()
            print("chat_id is none, nothing sent!")
            return
        data = {
            "chat_id": self._chat_id,
            "text": msg
        }
        if self._parse_mode:
            data["parse_mode"] = self._parse_mode
        try:
            response = requests.get(f"https://api.telegram.org/bot{self._token}/sendMessage", data=data, timeout=10)
            if response.status_code != 200 or response.json()["ok"] is not True:
                print(f"Failed to send notification:\n\tstatus_code={response.status_code}\n\tjson:\n\t{response.json()}")
        except Exception as e:
            print(f"Failed to send notification:\n\texception:\n\t{e}")


def setup_notifier_if_possible(token=None, token_path=None, token_env_var=None, chat_id=None, disable=False):
    send_notification = lambda message: None
    if disable:
        return send_notification
    try:
        import os
        kwargs = {}
        if chat_id:
            kwargs["chat_id"] = chat_id
        
        token_path_content = None
        if token_path:
            try:
                with open(filepath,'r') as f:
                    token_path_content = f.read()
            except:
                token_path_content = None
            
        notifier = TelegramNotifier(token=token or token_path_content or (token_env_var and os.environ.get(token_env_var)),  parse_mode="HTML", **kwargs)
        send_notification = lambda message: notifier.send(message)
    except Exception as error:
        print(f'''This isnt a big deal; but note there won't be any notifications because there was an error with the notifier: {error}''')
    
    return send_notification