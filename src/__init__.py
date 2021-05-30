__all__ = ["fitness", "gaparsimony", "ordenacion", "population"]

from .gaparsimony import GAparsimony
from .population import Population, Cromosoma
from .ordenacion import order
from .fitness import getFitness