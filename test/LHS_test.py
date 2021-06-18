# -*- coding: utf-8 -*-

from GAparsimony.lhs.util import isValidLHS, isValidLHS_int
from GAparsimony.lhs.base import geneticLHS, improvedLHS, maximinLHS, optimumLHS, randomLHS

import pytest

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_randomLHS_int(shape):
    assert isValidLHS_int(randomLHS.randomLHS_int(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_randomLHS(shape):
    assert isValidLHS(randomLHS.randomLHS(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_geneticLHS(shape):
    assert isValidLHS(geneticLHS.geneticLHS(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_improvedLHS(shape):
    assert isValidLHS(improvedLHS.improvedLHS(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_maximinLHS(shape):
    assert isValidLHS(maximinLHS.maximinLHS(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (8, 8)
])
def test_optimumLHS(shape):
    assert isValidLHS(optimumLHS.optimumLHS(*shape))