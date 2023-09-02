# What is this?

Simple automated tests; for [grug](https://grugbrain.dev/) devs like me who don't have time to test.

# How do I use this?

`pip install grug_test`


In your `main.py` equivlent:

```python
from grug_test import GrugTest
import os

# 1. say where the tests will be saved
grug_test = GrugTest(
    project_folder=".",
    test_folder="./tests/grug_tests",
    fully_disable=os.environ.get("PROD")!=None,
    replay_inputs=os.environ.get("RUN_TEST_CASES")!=None,
    record_io=os.environ.get("RECORD")!=None,
)

# 2. slap @grug_test on any of your pure-functions
@grug_test(max_io=100)
def repeat(a,times):
    for _ in range(times):
        a += f"{a}"
    return a

```

3. That's all the setup!
- `RECORD=True         python ./main.py` will record tests for you
- `RUN_TEST_CASES=True python ./main.py` will check your functions
- `PROD=True           python ./main.py` will run with grug totally disabled
- Make sure to commit the generated tests to git
    - When you do `RUN_TEST_CASES=True`, the git-diff will show you any problems
    - If you like the changes, well ✨volia✨ the git changes are your freshly-written test cases
    - If you don't like the changes, well then it looks like you've got some dev work to do
    - thats it

# Q&A

Does this work with `@staticmethod`?

- Yes but you have to put the decorator it on the line BELOW the `@staticmethod`

Does this work with ANY pure function?

- Almost, the arguments need to be seralizable. For example, if you pass a lambda function as an argument then grug_test can't really save/load that lambda function when `replay_inputs=True`. However, you can make almost any normal class seralizable, just checkout a tutorial on making a class work with python-pickle, or (even better) do `from grug_test import yaml` and make your class be yaml-seralizable (tutorial/example [here](https://github.com/jeff-hykin/ez_yaml/blob/8b4dce8bf495484feb50f84468ffc6f776c357d4/README.md#custom-yaml-tags-example))
