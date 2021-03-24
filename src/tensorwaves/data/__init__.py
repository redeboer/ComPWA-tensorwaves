# cspell:ignore Källén kallen
"""The `.data` module takes care of data generation."""

import logging
import math
from typing import Optional, Sequence, Tuple, TypeVar

import numpy as np
from ampform.data import EventCollection
from ampform.kinematics import ReactionInfo
from tqdm.auto import tqdm

from tensorwaves.data.phasespace import (
    TFPhaseSpaceGenerator,
    TFUniformRealNumberGenerator,
)
from tensorwaves.interfaces import (
    DataTransformer,
    Function,
    PhaseSpaceGenerator,
    UniformRealNumberGenerator,
)

from . import phasespace, transform

__all__ = [
    "generate_data",
    "generate_phsp",
    "phasespace",
    "transform",
]


def _generate_data_bunch(
    bunch_size: int,
    phsp_generator: PhaseSpaceGenerator,
    random_generator: UniformRealNumberGenerator,
    intensity: Function,
    kinematics: DataTransformer,
) -> Tuple[EventCollection, float]:
    phsp_sample, weights = phsp_generator.generate(
        bunch_size, random_generator
    )
    momentum_pool = EventCollection(phsp_sample)
    dataset = kinematics.transform(momentum_pool)
    intensities = intensity(dataset)
    maxvalue: float = np.max(intensities)

    uniform_randoms = random_generator(bunch_size, max_value=maxvalue)

    hit_and_miss_sample = momentum_pool.select_events(
        weights * intensities > uniform_randoms
    )
    return hit_and_miss_sample, maxvalue


def generate_data(
    size: int,
    reaction_info: ReactionInfo,
    data_transformer: DataTransformer,
    intensity: Function,
    phsp_generator: Optional[PhaseSpaceGenerator] = None,
    random_generator: Optional[UniformRealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> EventCollection:
    """Facade function for creating data samples based on an intensities.

    Args:
        size: Sample size to generate.
        reaction_info: Reaction info that is needed to define the phase space.
        data_transformer: An instance of `.DataTransformer` that is used to
            transform a generated `.DataSample` to a `.DataSample` that can be
            understood by the `.Function`.
        intensity: The intensity `.Function` that will be sampled.
        phsp_generator: Class of a phase space generator.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with **indeterministic** behavior.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    if phsp_generator is None:
        phsp_gen_instance = TFPhaseSpaceGenerator()
    phsp_gen_instance.setup(reaction_info)
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()

    progress_bar = tqdm(
        total=math.ceil(size / bunch_size),
        desc="Generating intensity-based sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    momentum_pool = EventCollection({})
    current_max = 0.0
    while momentum_pool.n_events < size:
        bunch, maxvalue = _generate_data_bunch(
            bunch_size,
            phsp_gen_instance,
            random_generator,
            intensity,
            data_transformer,
        )
        if maxvalue > current_max:
            current_max = 1.05 * maxvalue
            if momentum_pool.n_events > 0:
                logging.info(
                    "processed bunch maximum of %s is over current"
                    " maximum %s. Restarting generation!",
                    maxvalue,
                    current_max,
                )
                momentum_pool = EventCollection({})
                progress_bar.update()
                continue
        if np.size(momentum_pool, 0) > 0:
            momentum_pool.append(bunch)
        else:
            momentum_pool = bunch
        progress_bar.update()
    progress_bar.close()
    return momentum_pool.select_events(slice(0, size))


def generate_phsp(
    size: int,
    reaction_info: ReactionInfo,
    phsp_generator: Optional[PhaseSpaceGenerator] = None,
    random_generator: Optional[UniformRealNumberGenerator] = None,
    bunch_size: int = 50000,
) -> EventCollection:
    """Facade function for creating (unweighted) phase space samples.

    Args:
        size: Sample size to generate.
        reaction_info: A `ampform.kinematics.ReactionInfo`
            needed for the `.PhaseSpaceGenerator.setup` of the phase space
            generator instanced.
        phsp_generator: Class of a phase space generator.
        random_generator: A uniform real random number generator. Defaults to
            `.TFUniformRealNumberGenerator` with **indeterministic** behavior.
        bunch_size: Adjusts size of a bunch. The requested sample size is
            generated from many smaller samples, aka bunches.

    """
    if phsp_generator is None:
        phsp_generator = TFPhaseSpaceGenerator()
    phsp_generator.setup(reaction_info)
    if random_generator is None:
        random_generator = TFUniformRealNumberGenerator()

    progress_bar = tqdm(
        total=size / bunch_size,
        desc="Generating phase space sample",
        disable=logging.getLogger().level > logging.WARNING,
    )
    momentum_pool = EventCollection({})
    while momentum_pool.n_events < size:
        phsp_sample, weights = phsp_generator.generate(
            bunch_size, random_generator
        )
        hit_and_miss_randoms = random_generator(bunch_size)
        bunch = EventCollection(phsp_sample).select_events(
            weights > hit_and_miss_randoms
        )

        if momentum_pool.n_events > 0:
            momentum_pool.append(bunch)
        else:
            momentum_pool = bunch
        progress_bar.update()
    progress_bar.close()
    return momentum_pool.select_events(slice(0, size))


_T = TypeVar("_T")


def compute_phsp_volume(
    s: _T, masses: Sequence[float], sample_size: int
) -> float:
    """Compute phase space volume for an arbitrary number of particles.

    Compute phasespace volume of momentum space for an arbitrary number of
    particles in the final state using **Riemann integration**.

    .. note:: An analytic solution exists only for the volume of the phasespace
        of two-particle decays.

    .. seealso:: `Lecture notes
        <http://theory.gsi.de/~knoll/Lecture-notes/1-kinematic.pdf>`_
    """
    if len(masses) < 2:
        raise ValueError("Need at least two masses")
    if len(masses) == 2:
        m_1 = masses[0]
        m_2 = masses[1]
        return 2 * np.pi * np.sqrt(_kallen_function(s, m_1 ** 2, m_2 ** 2)) / s

    masses_new = list(masses)  # shallow copy with pop method
    s_prime = masses_new.pop() ** 2
    s_lower, s_upper = __create_s_range(s, masses_new)
    integration_sample = np.linspace(
        s_lower, s_upper, num=sample_size, endpoint=False
    )
    if np.isnan(integration_sample).any():
        raise ValueError(integration_sample)
    previous_phsp_volume = compute_phsp_volume(
        s=integration_sample,
        masses=masses_new,
        sample_size=sample_size,
    )
    if np.isnan(previous_phsp_volume).any():
        raise ValueError(previous_phsp_volume)
    previous_phsp = _kallen_function(s, integration_sample, s_prime)
    assert previous_phsp.shape == (sample_size,)
    previous_phsp = np.sqrt(previous_phsp)
    assert previous_phsp.shape == (sample_size,)
    previous_phsp *= previous_phsp_volume
    raise ValueError(previous_phsp.min())
    assert previous_phsp.shape == (sample_size,)
    bin_size = (s_lower - s_upper) / sample_size
    volume = np.sum(previous_phsp * bin_size) * np.pi / s
    return volume


def compute_phsp_volume_mc(
    s: _T, masses: Sequence[float], sample_size: int
) -> float:
    """Compute phase space volume for an arbitrary number of particles.

    Compute phasespace volume of momentum space for an arbitrary number of
    particles in the final state using **MC integration**.
    """
    if len(masses) < 2:
        raise ValueError("Need at least two masses")
    if len(masses) == 2:
        m_1 = masses[0]
        m_2 = masses[1]
        return 2 * np.pi * np.sqrt(_kallen_function(s, m_1 ** 2, m_2 ** 2)) / s


def _kallen_function(x: _T, y: _T, z: _T) -> _T:
    """Källén function.

    Original `Källén function
    <https://en.wikipedia.org/wiki/K%C3%A4ll%C3%A9n_function>`_, that is, not
    having square values in its argument. We use this function instead of the
    one that can be factorized (see `Heron's formula
    <https://en.wikipedia.org/wiki/Heron%27s_formula>`_), because we need to
    enter :math:`s` without taking its square root.
    """
    return x ** 2 + y ** 2 + z ** 2 - 2 * x * y - 2 * y * z - 2 * z * x  # type: ignore


def __create_s_range(s: _T, masses: Sequence[float]) -> Tuple[float, float]:
    total_mass = sum(masses)
    s_lower = total_mass ** 2
    s_upper = np.sqrt(s) - masses[-1]
    s_upper *= s_upper
    return s_lower, s_upper
