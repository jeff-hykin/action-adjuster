from grug_test import GrugTest
import os

grug_test = GrugTest(
    project_folder=".",
    test_folder="./tests/grug_tests",
    fully_disable=os.environ.get("PROD")!=None,
    record_io=os.environ.get("DEV")!=None,
)

@grug_test
def add_nums(a,b):
    return a + b + 1

# normal usage
for a,b in zip(range(10), range(30, 40)):
    add_nums(a,b)