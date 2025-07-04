"""
Config values from the original macbf, left out TOP_K and dynamic matrices
"""
TIME_STEP = 0.01
TIME_STEP_EVAL = 0.01
OBS_RADIUS = 1.0

DIST_MIN_THRES = 0.8
DIST_MIN_CHECK = 0.6
DIST_SAFE = 1.0
DIST_TOLERATE = 0.7

ALPHA_CBF = 1.0
WEIGHT_DECAY = 1e-8

TRAIN_STEPS = 70000
EVALUATE_STEPS = 5
INNER_LOOPS = 40
INNER_LOOPS_EVAL = 40
DISPLAY_STEPS = 5
SAVE_STEPS = 5

LEARNING_RATE = 1e-4
REFINE_LEARNING_RATE = 1.0
REFINE_LOOPS = 40