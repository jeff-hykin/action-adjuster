import os

from generic_tools.notifier import setup_notifier_if_possible
from __dependencies__.quik_config import find_and_load
from __dependencies__.blissful_basics import FS, LazyDict, Warnings
from __dependencies__.grug_test import GrugTest
from __dependencies__.telepy_notify import Notifier

# Warnings.disable()

info = find_and_load(
    "main/config.yaml",
    cd_to_filepath=True,
    fully_parse_args=True,
    defaults_for_local_data=[ "WARTHOG" ],
)

absolute_path_to  = info.absolute_path_to
path_to           = info.path_to
config            = info.config
secrets           = info.secrets
selected_profiles = list(info.selected_profiles)
debug = LazyDict()

config.output_folder = config.get("output_folder", path_to.default_output_folder)
# stamp some things
FS.copy(item="config.yaml", to=config.output_folder, new_name=None)
import subprocess
commit_hash = subprocess.check_output(['git', 'rev-parse', "HEAD"]).decode('utf-8')[0:-1]
FS.write(data=commit_hash, to=f"{config.output_folder}/commit_hash.log")

project_folder = FS.parent_path(FS.parent_path(absolute_path_to.main))
grug_test = GrugTest(
    project_folder=project_folder,
    test_folder=f"{project_folder}/tests/grug_tests",
    fully_disable=config.grug_test.disable,
    replay_inputs=config.grug_test.replay_inputs or os.getenv("GRUG_TEST"),
    record_io=config.grug_test.record_io or os.getenv("GRUG_RECORD"),
    verbose=config.grug_test.verbose,
)

send_notification = setup_notifier_if_possible(
    disable=not secrets.get("send_notification", False),
    token=secrets.get("telegram_token", None),
    chat_id=secrets.get("telegram_chat_id", None),
)

notifier = Notifier(
    disable=not secrets.get("send_notification", False),
    token=secrets.get("telegram_token", None),
    chat_id=secrets.get("telegram_chat_id", None),
)