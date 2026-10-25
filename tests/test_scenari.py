import numpy as np
import pytest
from scipy import stats

from wealth_planner import Mercato, genera
from wealth_planner.scenari import parametri_lognormali


def test_parametri_lognormali_riproducono_media_e_volatilita():
    mu, sigma = parametri_lognormali(0.07, 0.16)
    media = np.exp(mu + sigma**2 / 2) - 1
    vol = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2))
    assert media == pytest.approx(0.07)
    assert vol == pytest.approx(0.16)


@pytest.mark.parametrize("distribuzione", ["normale", "regimi"])
def test_rendimenti_simulati_rispettano_le_ipotesi(distribuzione):
    m = Mercato(distribuzione=distribuzione)
    sc = genera(m, anni=40, n=10_000, seed=1)
    r = sc.rendimenti.reshape(-1, 3)
    for i, asset in enumerate(m.asset.values()):
        assert r[:, i].mean() == pytest.approx(asset.rendimento, abs=0.003)
        assert r[:, i].std() == pytest.approx(asset.volatilita, rel=0.03)
    corr = np.corrcoef(np.log1p(r).T)
    assert corr[0, 1] == pytest.approx(m.correlazioni[0][1], abs=0.02)
    assert corr[1, 2] == pytest.approx(m.correlazioni[1][2], abs=0.02)


def test_regimi_producono_code_piu_grasse():
    normale = genera(Mercato(distribuzione="normale"), 40, 10_000, seed=2).rendimenti[..., 0].ravel()
    regimi = genera(Mercato(distribuzione="regimi"), 40, 10_000, seed=2).rendimenti[..., 0].ravel()
    assert stats.kurtosis(np.log1p(regimi)) > stats.kurtosis(np.log1p(normale)) + 0.5
    assert np.percentile(regimi, 1) < np.percentile(normale, 1)


def test_inflazione_ar1_media_e_persistenza():
    m = Mercato(inflazione_media=0.025, inflazione_volatilita=0.012, inflazione_persistenza=0.6)
    infl = genera(m, 50, 5_000, seed=3).inflazione
    assert infl.mean() == pytest.approx(0.025, abs=0.001)
    assert infl.std() == pytest.approx(0.012, rel=0.05)
    autocorr = np.corrcoef(infl[:, :-1].ravel(), infl[:, 1:].ravel())[0, 1]
    assert autocorr == pytest.approx(0.6, abs=0.03)


def test_stesso_seme_stessi_scenari():
    a, b = genera(Mercato(), 10, 100, seed=7), genera(Mercato(), 10, 100, seed=7)
    np.testing.assert_array_equal(a.rendimenti, b.rendimenti)


def test_correlazioni_non_valide():
    with pytest.raises(ValueError):
        Mercato(correlazioni=((1, 0.9, 0.9), (0.9, 1, -0.9), (0.9, -0.9, 1)))
    with pytest.raises(ValueError):
        Mercato(correlazioni=((1, 0), (0, 1)))
