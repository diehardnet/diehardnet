"""Hans Gruber training noise injector
This file encapsulate the noise injector to be used
in the training process
"""
import random
import warnings

import numpy as np
import torch

LINE, SQUARE, RANDOM, ALL = "LINE", "SQUARE", "RANDOM", "ALL"


class HansGruberNI(torch.nn.Module):
    def __init__(self, error_model: str = LINE):
        super(HansGruberNI, self).__init__()
        # Error model necessary for the forward
        self.error_model = error_model
        self.noise_data = list()

    def set_noise_data(self, noise_data: list = None) -> None:
        r"""Set the noise data that we extract and parse from radiation experiments
        The noise data is extracted form a CSV file, pass only a numpy array to the function
        """
        # make a subset of the errors
        self.noise_data = [i for i in noise_data if i["geometry_format"] == self.error_model]

    def forward(self, forward_input: torch.Tensor) -> torch.Tensor:
        r"""Perform a 'forward' operation to simulate the error model injection
        in the training process
        :param forward_input: torch.Tensor input for the forward
        :return: processed torch.Tensor
        """
        # TODO: How to inject the error model? Is it static for the whole training?
        #  I believe we should randomize it, let's say: we pick a given
        #  layer and at each forward we randomly sample a certain feature
        #  map to corrupt among all the features

        # We can inject the relative errors using only Torch built-in functions
        # Otherwise it is necessary to use AutoGrads
        output = forward_input.clone()
        # TODO: It must be generalized for tensors that have more than 2 dim
        warnings.warn("Need to fix the HansGruber noise injector to support more than 2d dimension before use")
        # rows, cols = input.shape
        relative_error = random.choice(self.noise_data)
        relative_error = random.uniform(float(relative_error["min_relative"]), float(relative_error["max_relative"]))
        if self.error_model == LINE:
            # relative_errors = torch.FloatTensor(1, rows).uniform_(0, 1)
            rand_row = random.randrange(0, forward_input.shape[0])
            output[rand_row, :].mul_(relative_error)
        # elif self.error_model == ErrorModel.COL:
        #     # relative_errors = torch.FloatTensor(1, cols).uniform_(0, 1)
        #     rand_col = random.randrange(0, input.shape[1])
        #     output[:, rand_col].mul_(relative_error)
        else:
            raise NotImplementedError
        print(forward_input[forward_input != output])

        return output
