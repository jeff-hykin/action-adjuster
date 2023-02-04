from quik_config import find_and_load

info = find_and_load(
    "main/config.yaml",
    cd_to_filepath=True,
    fully_parse_args=True,
    defaults_for_local_data=[],
)

path_to = info.path_to
config  = info.config