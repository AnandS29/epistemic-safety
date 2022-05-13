from general_example import *

risk_levels_to_run = [0, 0.1, 1.0, 2.0, 3.0, 5.0]
for state in game_types:
    for alpha in abs_util:
        for risk in risk_levels_to_run:
            print("Experiment:", state, alpha, risk)
            run_exp(state, alpha, risk)