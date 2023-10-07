#!/usr/bin/env python3

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

from enum import Enum


class GammaTypes(Enum):
    PROB = 1,
    LOG_PROB = 2,
    LOGIT = 3
