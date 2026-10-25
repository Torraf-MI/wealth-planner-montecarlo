"""Generazione degli scenari di mercato: rendimenti annui correlati delle classi di attività e inflazione."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modello import Mercato


@dataclass(frozen=True)
class Scenari:
    """Traiettorie simulate. Gli stessi scenari si possono riusare per confrontare piani diversi a parità di mercato."""

    rendimenti: np.ndarray  # (simulazioni, anni, classi): rendimenti annui semplici nominali
    inflazione: np.ndarray  # (simulazioni, anni): inflazione annua
    crisi: np.ndarray  # (simulazioni, anni): True negli anni turbolenti
    asset: tuple[str, ...]

    @property
    def n(self) -> int:
        return self.rendimenti.shape[0]

    @property
    def anni(self) -> int:
        return self.rendimenti.shape[1]


def parametri_lognormali(media: np.ndarray | float, volatilita: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Media e deviazione standard del logaritmo di 1 + r, dati media e volatilità del rendimento semplice r."""
    media, volatilita = np.asarray(media, dtype=float), np.asarray(volatilita, dtype=float)
    sigma2 = np.log1p(volatilita**2 / (1 + media) ** 2)
    return np.log1p(media) - sigma2 / 2, np.sqrt(sigma2)


def inflazione_ar1(mercato: Mercato, anni: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Inflazione annua come processo AR(1) attorno alla media, partendo dalla distribuzione stazionaria."""
    m, s, phi = mercato.inflazione_media, mercato.inflazione_volatilita, mercato.inflazione_persistenza
    eps = rng.standard_normal((n, anni))
    infl = np.empty((n, anni))
    infl[:, 0] = m + s * eps[:, 0]
    s_innovazione = s * np.sqrt(1 - phi**2)
    for t in range(1, anni):
        infl[:, t] = m + phi * (infl[:, t - 1] - m) + s_innovazione * eps[:, t]
    return infl


def genera(mercato: Mercato, anni: int, n: int = 5_000, seed: int | None = 42) -> Scenari:
    """Simula ``n`` traiettorie di ``anni`` anni.

    I rendimenti sono lognormali correlati (fattorizzazione di Cholesky della matrice delle correlazioni). Con la
    distribuzione a regimi la volatilità degli anni normali viene ridotta in modo che la volatilità complessiva
    coincida con quella indicata, e la media di ogni regime è corretta in modo che il rendimento atteso non cambi.
    """
    rng = np.random.default_rng(seed)
    asset = list(mercato.asset.values())
    media = np.array([a.rendimento for a in asset])
    vol = np.array([a.volatilita for a in asset])
    chol = np.linalg.cholesky(np.asarray(mercato.correlazioni, dtype=float) + 1e-12 * np.eye(len(asset)))
    z = rng.standard_normal((n, anni, len(asset))) @ chol.T

    if mercato.distribuzione == "regimi":
        p, k = mercato.prob_crisi, mercato.moltiplicatore_crisi
        crisi = rng.random((n, anni)) < p
        vol_calma = vol / np.sqrt(1 - p + p * k**2)
        mu_c, sig_c = parametri_lognormali(media, vol_calma)
        mu_t, sig_t = parametri_lognormali(media, vol_calma * k)
        mu = np.where(crisi[..., None], mu_t, mu_c)
        sigma = np.where(crisi[..., None], sig_t, sig_c)
    else:
        crisi = np.zeros((n, anni), dtype=bool)
        mu, sigma = parametri_lognormali(media, vol)

    rendimenti = np.expm1(mu + sigma * z)
    return Scenari(rendimenti=rendimenti, inflazione=inflazione_ar1(mercato, anni, n, rng), crisi=crisi,
                   asset=mercato.chiavi)
