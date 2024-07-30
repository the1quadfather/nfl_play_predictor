# nfl_play_predictor
NFL next-play predictor based on a set of variables from available play-by-play data, 2013-present.

The two ipynb files are the primary files to use the tool as-is.
nfl_pbp_predictor_sgd_cpd.ipynb runs the predictor code in a loop using a stored Joblib-formatted model checkpoint. (A pre-ran checkpoint is available by downloading or pointing to the nfl_pbp_sgdclass.joblib file)
playcall_predictor_sgdclass.ipynb runs the entire training code and allows you to save a Joblib model checkpoint at the end.

For system ease-of-use and agnostics, these files were created and intended to be run on Colab, although they can easily be run using localy Jupyter kernals if one would rather.

With the current dataset and variable setup, the SGD-enhanced classifier that performs the prediciton has a test accuracy of 65%.

Future work includes enhancing the dataset via formatting and different variable coding, and optimizing a vanilla and/or recursive neural network on this task and comparing the results.

Created using Google Colab, Scikit Learn, and NFL Savant's play-by-play data. All libraries, APIs, and data references should be considered attributed to their respective owners.
Any references to this repository, code, or linked code should be attributed to Schraeder Technologies.
