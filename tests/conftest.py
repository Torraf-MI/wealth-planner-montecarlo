import pytest

from wealth_planner import AssetClass, Cliente, Flussi, Mercato, Piano, Portafoglio


def mercato_deterministico(rendimento: float = 0.05, inflazione: float = 0.0) -> Mercato:
    """Un'unica attività senza volatilità e inflazione costante: i risultati si verificano a mano."""
    return Mercato(asset={"fondo": AssetClass("Fondo", rendimento=rendimento, volatilita=0.0)},
                   correlazioni=((1.0,),), inflazione_media=inflazione, inflazione_volatilita=0.0,
                   distribuzione="normale")


def piano_deterministico(rendimento: float = 0.05, inflazione: float = 0.0, **flussi) -> Piano:
    flussi.setdefault("patrimonio_iniziale", 100_000)
    return Piano(
        cliente=Cliente(eta=60, eta_pensione=60, eta_finale=80),
        flussi=Flussi(**flussi),
        portafoglio=Portafoglio(pesi={"fondo": 1.0}, costi=0.0, bollo=0.0, aliquota_plusvalenze=0.0),
        mercato=mercato_deterministico(rendimento, inflazione),
    )


@pytest.fixture
def deterministico():
    return piano_deterministico
