import random
from contextlib import contextmanager
from enum import IntEnum

import numpy as np
import torch


class RandomStream(IntEnum):
    GP_TRAIN_INPUTS = 0
    GP_EVAL_INPUTS = 1
    GP_OBSERVATIONS = 2
    DIRECTIONS = 3
    BO_STEP = 4
    TRAINING = 5
    ACQUISITION = 6
    OBJECTIVE_NOISE = 7
    TURBO_CANDIDATES = 8
    TURBO_MASK = 9
    TURBO_POSTERIOR = 10
    TURBO_RESTART = 11
    TRAINING_BATCH = 12
    DERIVATIVE_SUBSET = 13
    INITIALIZATION = 14
    ACQUISITION_MC = 15


def seed_for(seed: int, stream: RandomStream, *indices: int) -> int:
    sequence = np.random.SeedSequence(
        int(seed), spawn_key=(int(stream), *(int(i) for i in indices))
    )
    return int(sequence.generate_state(1)[0])


@contextmanager
def rng_scope(seed: int, *, device: torch.device | str = "cpu"):
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        devices = [torch.cuda.current_device() if device.index is None else device.index]
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    with torch.random.fork_rng(devices=devices):
        try:
            torch.default_generator.manual_seed(int(seed))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(seed))
            np.random.seed(np.random.SeedSequence(int(seed)).generate_state(1))
            random.seed(int(seed))
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
