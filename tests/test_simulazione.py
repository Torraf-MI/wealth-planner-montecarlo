import numpy as np
import pytest

from wealth_planner import Cliente, Uscita, esempio_rossi, simula
from wealth_planner.simulazione import pesi_annui


def test_accumulo_con_versamenti_formula_chiusa(deterministico):
    p = deterministico(patrimonio_iniziale=50_000, versamento_annuo=10_000)
    p = p.con(cliente=Cliente(eta=40, eta_pensione=60, eta_finale=61))
    r = simula(p, n=3)
    atteso = 50_000 * 1.05**20 + 10_000 * sum(1.05**k for k in range(1, 21))
    assert r.patrimonio[0, r.colonna(60)] == pytest.approx(atteso)


def test_esaurimento_nell_anno_previsto(deterministico):
    # 100.000 € a rendimento nullo, 10.000 € l'anno: bastano per 10 prelievi, l'undicesimo manca
    p = deterministico(rendimento=0.0, spesa_annua=10_000)
    r = simula(p, n=2)
    assert r.esaurito.all()
    assert r.eta_esaurimento[0] == 70
    assert r.prelievi[0, :10].sum() == pytest.approx(100_000)


def test_pensione_pubblica_riduce_il_prelievo(deterministico):
    p = deterministico(rendimento=0.0, spesa_annua=30_000, pensione_pubblica=25_000)
    r = simula(p, n=2)
    assert r.prelievi[0, 0] == pytest.approx(5_000)
    assert not r.esaurito.any()


def test_inflazione_rivaluta_i_prelievi(deterministico):
    p = deterministico(rendimento=0.0, inflazione=0.02, patrimonio_iniziale=1_000_000, spesa_annua=10_000)
    r = simula(p, n=2)
    np.testing.assert_allclose(r.prelievi[0, :5], 10_000 * 1.02 ** np.arange(5))
    np.testing.assert_allclose(r.prelievi_reali[0], 10_000)


def test_imposta_sulle_plusvalenze_costo_medio(deterministico):
    # valore 200.000, carico 100.000: metà di ogni prelievo è plusvalenza, aliquota effettiva 13%
    p = deterministico(rendimento=0.0, patrimonio_iniziale=200_000, costo_fiscale_iniziale=100_000,
                       spesa_annua=87_000)
    p = p.con(portafoglio__aliquota_plusvalenze=0.26)
    r = simula(p, n=2)
    assert r.imposte[0, 0] == pytest.approx(13_000)
    assert r.patrimonio[0, 1] == pytest.approx(100_000)
    # il carico si riduce in proporzione: la quota di plusvalenza resta 50%
    assert r.imposte[0, 1] / (r.imposte[0, 1] + r.prelievi[0, 1]) == pytest.approx(0.13)


def test_costi_e_bollo(deterministico):
    p = deterministico(rendimento=0.05).con(portafoglio__costi=0.01, portafoglio__bollo=0.002)
    r = simula(p, n=2)
    assert r.patrimonio[0, 1] == pytest.approx(100_000 * 1.04 * 0.998)


def test_uscita_straordinaria(deterministico):
    p = deterministico(rendimento=0.0).con(uscite=(Uscita(eta=65, importo=30_000),))
    r = simula(p, n=2)
    assert r.patrimonio[0, r.colonna(66)] == pytest.approx(70_000)


def test_strategia_percentuale_non_esaurisce_mai():
    p = esempio_rossi().con(flussi__strategia="percentuale", uscite=())
    assert not simula(p, n=1_000).esaurito.any()


def test_guardrail_migliora_la_probabilita_rispetto_al_fisso():
    p = esempio_rossi()
    fisso = simula(p, n=2_000)
    guard = simula(p.con(flussi__strategia="guardrail"), n=2_000)
    assert guard.esaurito.mean() < fisso.esaurito.mean()
    assert guard.fattore_spesa.min() >= 1 - p.flussi.taglio_massimo - 1e-12
    assert guard.fattore_spesa.max() <= 1.0


def test_percorso_life_cycle():
    p = esempio_rossi()
    w = pesi_annui(p)
    np.testing.assert_allclose(w.sum(axis=1), 1)
    assert w[0, 0] == pytest.approx(0.70)
    assert w[p.cliente.eta_pensione - p.cliente.eta, 0] == pytest.approx(0.45)
    assert w[-1, 0] == pytest.approx(0.45)
