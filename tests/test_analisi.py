import numpy as np
import pytest

from wealth_planner import (
    Cliente,
    esempio_rossi,
    genera,
    griglia,
    prelievo_sostenibile,
    probabilita_obiettivo,
    probabilita_successo,
    simula,
    versamento_necessario,
)
from wealth_planner.analisi import percentili, rischio_longevita, sopravvivenza
from wealth_planner.modello import pesi_con_quota_azionaria


@pytest.fixture(scope="module")
def base():
    p = esempio_rossi()
    return p, genera(p.mercato, p.anni, 2_000, seed=11)


def test_prelievo_sostenibile_raggiunge_la_probabilita(base):
    p, sc = base
    spesa = prelievo_sostenibile(p, 0.90, sc)
    assert probabilita_successo(simula(p.con(flussi__spesa_annua=spesa - 100), sc)) >= 0.90
    assert probabilita_successo(simula(p.con(flussi__spesa_annua=spesa + 200), sc)) < 0.90


def test_versamento_necessario_per_l_obiettivo(base):
    p, sc = base
    v = versamento_necessario(p, 0.80, scenari=sc)
    assert v > p.flussi.versamento_annuo  # oggi l'obiettivo è raggiunto in meno di 8 casi su 10
    assert probabilita_obiettivo(simula(p.con(flussi__versamento_annuo=v + 100), sc)) >= 0.80
    assert versamento_necessario(p.con(obiettivo__importo=100_000), 0.80, scenari=sc) == 0.0


def test_griglia_decrescente_nella_spesa(base):
    p, sc = base
    g = griglia(p, spese=[30_000, 38_000, 46_000], quote=[0.3, 0.7], scenari=sc)
    assert (np.diff(g.to_numpy(), axis=0) <= 0).all()


def test_percentili_ordinati(base):
    p, sc = base
    df = percentili(simula(p, sc))
    assert (np.diff(df.to_numpy(), axis=1) >= 0).all()
    assert df.loc[p.cliente.eta, "P50"] == pytest.approx(p.flussi.patrimonio_iniziale)


def test_sopravvivenza_gompertz():
    c = Cliente(eta=65)
    s = sopravvivenza(c, [65, 80, 90, 100])
    assert s[0] == pytest.approx(1)
    assert (np.diff(s) < 0).all()
    assert 0.3 < s[2] < 0.6  # circa metà dei sessantacinquenni arriva a 90 anni con età modale 90


def test_rischio_longevita_inferiore_al_rischio_di_esaurimento(base):
    p, sc = base
    r = simula(p, sc)
    assert 0 < rischio_longevita(r) < r.esaurito.mean()


def test_quota_azionaria():
    p = esempio_rossi()
    pesi = pesi_con_quota_azionaria(p.portafoglio.pesi, p.mercato, 0.40)
    assert sum(pesi.values()) == pytest.approx(1)
    assert pesi["azioni"] == pytest.approx(0.40)
    assert pesi["obbligazioni"] / pesi["liquidita"] == pytest.approx(0.25 / 0.05)
    assert p.con_quota_azionaria(0.4).portafoglio.pesi_pensione is None
