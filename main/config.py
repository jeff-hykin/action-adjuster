from quik_config import find_and_load
import file_system_py as FS

info = find_and_load(
    "main/config.yaml",
    cd_to_filepath=True,
    fully_parse_args=True,
    defaults_for_local_data=[ "WARTHOG" ],
)

path_to = info.path_to
config  = info.config

# stamp some things
FS.copy(item="config.yaml", to=path_to.default_output_folder, new_name=None)
import subprocess
commit_hash = subprocess.check_output(['git', 'rev-parse', "HEAD"]).decode('utf-8')[0:-1]
FS.write(data=commit_hash, to=f"{path_to.default_output_folder}/commit_hash.log")