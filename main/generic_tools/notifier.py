def setup_notifier_if_possible(token=None, token_path=None, token_env_var=None, chat_id=None,):
    send_notification = lambda message: None
    try:
        import os
        from telegram_notifier import TelegramNotifier
        kwargs = {}
        if chat_id:
            kwargs["chat_id"] = chat_id
        notifier = TelegramNotifier(token=token or FS.read(token_path) or (token_env_var and os.environ.get(token_env_var)),  parse_mode="HTML", **kwargs)
        send_notification = lambda message: notifier.send(message)
    except Exception as error:
        print(f'''This isnt a big deal; but note there won't be any notifications because there was an error with the notifier: {error}''')
    
    return send_notification