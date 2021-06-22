# -*- coding: utf-8 -*-

from GAparsimony.lhs.util import isValidLHS, isValidLHS_int
from GAparsimony.lhs import geneticLHS, improvedLHS, maximinLHS, optimumLHS, randomLHS, randomLHS_int

import pytest

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_randomLHS_int(shape):
    assert isValidLHS_int(randomLHS_int(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_randomLHS(shape):
    assert isValidLHS(randomLHS(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_geneticLHS(shape):
    assert isValidLHS(geneticLHS(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_improvedLHS(shape):
    assert isValidLHS(improvedLHS(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (3, 8)
])
def test_maximinLHS(shape):
    assert isValidLHS(maximinLHS(*shape))

@pytest.mark.parametrize("shape", [
    (2, 2),
    (6, 6),
    (8, 8)
])
def test_optimumLHS(shape):
    assert isValidLHS(optimumLHS(*shape))