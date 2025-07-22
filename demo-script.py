MODE = "FAST"

if MODE == "FAST":
    from algorithme15 import make_population, my_graphic_function, my_function, get_feature_importance, datapoint_to_csv
else:
    from algorithme90 import make_population, my_graphic_function, my_function, get_feature_importance, datapoint_to_csv
from random import choice
import json
TARGET_FILE = "../data-room/backtest.csv"

POP = make_population(TARGET_FILE)

X = choice(POP)
print("### MY GRAPHIC FUNCTION ###")
print(json.dumps(my_graphic_function(X), indent=2))
print("### MY FUNCTION ###")
print(my_function(X))
print("### GET FEATURE IMPORTANCE ###")
print(get_feature_importance(X))
print("### DATAPOINT TO CSV ###")
print(datapoint_to_csv(X))

