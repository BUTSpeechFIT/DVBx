#!/usr/bin/env python3

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

from enum import Enum


class AvailableBackpropTypes(Enum):
    AFTER_CONVERGENCE = 1
    AFTER_EACH_ITER = 2
