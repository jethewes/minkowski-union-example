# minkowski-union-example
This is a minimal example for reproducing a backprop error in the `MinkowskiUnion` layer in `MinkowskiEngine`. The model architecture is included in `model.py`, a single example input is included in `data/example.pt`, and the script `example.py` should reproduce the error when executed with Python.

This network was developed for a particle physics application, in which two independent pixel map representations are used simultaneously to classify an event. The two pixel maps are independently convolved, and then the Union function is used to combine them before a final set of convolutions.

Note: This example _also_ includes an error in the `MinkowskiChannelwiseConvolution` module. This layer throws an error when included in the model, but since it convolves in place without changing the dimensionality of the tensor, it has been commented out in the model file – this second error can be reproduced by uncommenting lines 40-44 of `model.py`.
