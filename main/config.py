from __dependencies__.quik_config import find_and_load
from __dependencies__.blissful_basics import FS, LazyDict

info = find_and_load(
    "main/config.yaml",
    cd_to_filepath=True,
    fully_parse_args=True,
    defaults_for_local_data=[ "WARTHOG" ],
)

absolute_path_to  = info.absolute_path_to
path_to           = info.path_to
config            = info.config
selected_profiles = list(info.selected_profiles)
debug = LazyDict()

config.output_folder = config.get("output_folder", path_to.default_output_folder)
# stamp some things
FS.copy(item="config.yaml", to=config.output_folder, new_name=None)
import subprocess
commit_hash = subprocess.check_output(['git', 'rev-parse', "HEAD"]).decode('utf-8')[0:-1]
FS.write(data=commit_hash, to=f"{config.output_folder}/commit_hash.log")