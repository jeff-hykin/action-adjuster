from elegant_events import Server

basic = Server("localhost", 7070, debugging=True)
# basic = Server("localhost", 7070)

@basic.whenever("test_point")
def yo(*args):
    print(f'''args = {args}''')
print("setup callback")

print("yelling callback")
basic.yell(event_name="test_point", data="howdy")
print("checking")
basic.check()