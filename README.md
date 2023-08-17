# Installation/Setup

Everything is detailed in the `documentation/setup.md`!


# Running Code

after doing `commands/start`

```sh
python ./main/main.py @WARTHOG
```

All the output (including the render images) is dumped into `main/output.ignore/`

To disable rendering:
```sh
python ./main/main.py @WARTHOG simulator:should_render:False
```

# Running experiments

1. Change the `experiment_number` inside the `config.yaml`
2. Run `python ./main/specific_tools/comparision_runs.py`
3. Once the experiment is finished, run `python ./main/plots/main_comparision.py` which will both display graphs and generate html files

# Options

As an example, open up `main/config.yaml` and find `max_velocity:`
To change that value through the CLI do:
```sh
python ./main/main.py @WARTHOG simulator:max_velocity:10
```

Similarly find `direct_velocity_cost:`. To change that through the command line would be:
```sh
python ./main/main.py @WARTHOG reward_parameters:direct_velocity_cost:10
```

Hopefully you can see the pattern; all values in the `config.yaml` can be overridden from the command line.