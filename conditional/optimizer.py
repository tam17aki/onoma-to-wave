# -*- coding: utf-8 -*-
"""A Python module which provides optimizer and scheduler.

Copyright (C) 2022 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from torch import optim


def get_optimizer(model, cfg):
    """Instantiate optimizer."""
    optimizer = {"seq2seq": None, "event": None}

    optimizer["seq2seq"] = optim.RAdam(
        model["seq2seq"].parameters(), lr=cfg.training.learning_rate
    )
    optimizer["event"] = optim.RAdam(
        model["event"].parameters(), lr=cfg.training.learning_rate
    )

    return optimizer


def get_scheduler(optimizer, cfg):
    """Instantiate scheduler."""
    scheduler = {"seq2seq": None, "evenet": None}

    scheduler["seq2seq"] = optim.lr_scheduler.MultiStepLR(
        optimizer["seq2seq"],
        milestones=cfg.training.milestones,
        gamma=cfg.training.gamma,
    )
    scheduler["event"] = optim.lr_scheduler.MultiStepLR(
        optimizer["event"],
        milestones=cfg.training.milestones,
        gamma=cfg.training.gamma,
    )

    return scheduler
