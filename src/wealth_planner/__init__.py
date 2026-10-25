"""Pianificazione patrimoniale Monte Carlo: obiettivi, sostenibilità dei prelievi in pensione, inflazione."""

from .analisi import (
    confronta_strategie,
    griglia,
    leve,
    percentili,
    prelievo_sostenibile,
    probabilita_obiettivo,
    probabilita_successo,
    rischio_longevita,
    sintesi,
    versamento_necessario,
)
from .modello import (
    AssetClass,
    Cliente,
    Flussi,
    Mercato,
    Obiettivo,
    Piano,
    Portafoglio,
    Uscita,
    esempio_rossi,
    load_toml,
)
from .scenari import Scenari, genera
from .simulazione import Risultato, simula

__all__ = [
    "AssetClass",
    "Cliente",
    "Flussi",
    "Mercato",
    "Obiettivo",
    "Piano",
    "Portafoglio",
    "Risultato",
    "Scenari",
    "Uscita",
    "confronta_strategie",
    "esempio_rossi",
    "genera",
    "griglia",
    "leve",
    "load_toml",
    "percentili",
    "prelievo_sostenibile",
    "probabilita_obiettivo",
    "probabilita_successo",
    "rischio_longevita",
    "simula",
    "sintesi",
    "versamento_necessario",
]

__version__ = "0.1.0"
