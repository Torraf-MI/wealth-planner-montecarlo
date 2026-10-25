"""Indicatori del piano: probabilità di successo e di obiettivo, prelievo sostenibile, sensitività, rischi."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .modello import Cliente, Piano
from .scenari import Scenari, genera
from .simulazione import Risultato, flussi_programmati, simula

PERCENTILI = (5, 25, 50, 75, 95)


def probabilita_successo(r: Risultato) -> float:
    """Quota di scenari in cui il patrimonio copre tutte le spese previste fino all'età finale."""
    return float(1 - r.esaurito.mean())


def probabilita_obiettivo(r: Risultato) -> float:
    """Quota di scenari in cui, all'età dell'obiettivo, il patrimonio reale raggiunge l'importo richiesto."""
    ob = r.piano.obiettivo
    if ob is None:
        return float("nan")
    return float((r.patrimonio_reale[:, r.colonna(ob.eta)] >= ob.importo).mean())


def percentili(r: Risultato, reale: bool = True, q: Sequence[int] = PERCENTILI) -> pd.DataFrame:
    """Percentili del patrimonio a inizio di ogni anno, per età (in euro di oggi se ``reale``)."""
    valori = r.patrimonio_reale if reale else r.patrimonio
    return pd.DataFrame(np.percentile(valori, q, axis=0).T, index=pd.Index(r.eta, name="Età"),
                        columns=[f"P{p}" for p in q])


def probabilita_patrimonio(r: Risultato) -> pd.Series:
    """Probabilità, per ogni età, che il piano stia ancora coprendo le spese (nessun ammanco fino ad allora)."""
    mancanza = np.zeros((r.n, len(r.eta)), dtype=bool)
    mancanza[:, 1:] = np.cumsum(r.ammanchi > 1.0, axis=1) > 0
    return pd.Series(1 - mancanza.mean(axis=0), index=pd.Index(r.eta, name="Età"), name="Piano in equilibrio")


def sopravvivenza(cliente: Cliente, eta: np.ndarray | Sequence[float]) -> np.ndarray:
    """Probabilità di essere in vita a ciascuna età, condizionata all'età attuale (legge di Gompertz)."""
    m, b = cliente.eta_modale_decesso, cliente.dispersione_decesso
    eta = np.asarray(eta, dtype=float)
    return np.exp(np.exp((cliente.eta - m) / b) - np.exp((eta - m) / b))


def rischio_longevita(r: Risultato) -> float:
    """Probabilità di restare senza risorse mentre si è ancora in vita.

    Combina l'età di esaurimento di ogni scenario con la probabilità di sopravvivere fino a quell'età
    (mortalità indipendente dai mercati). È più bassa della probabilità di esaurimento entro l'età finale,
    perché in molti scenari il patrimonio finisce a un'età che il cliente potrebbe non raggiungere.
    """
    eta = r.eta_esaurimento
    s = np.where(np.isnan(eta), 0.0, sopravvivenza(r.piano.cliente, np.nan_to_num(eta, nan=0.0)))
    return float(s.mean())


def rischio_sequenza(r: Risultato, anni: int = 10) -> pd.DataFrame:
    """Rendimento reale annualizzato nei primi ``anni`` di pensione ed esito di ogni scenario.

    Mostra il rischio di sequenza: a parità di rendimento medio di lungo periodo, anni negativi subito dopo
    la pensione, quando si inizia a prelevare, compromettono il piano molto più che a fine percorso.
    """
    inizio = r.colonna(r.piano.cliente.eta_pensione) if r.piano.cliente.eta_pensione < r.eta[-1] else 0
    fine = min(inizio + anni, r.rendimenti.shape[1])
    reale = (1 + r.rendimenti[:, inizio:fine]) / (1 + r.inflazione[:, inizio:fine])
    annuo = reale.prod(axis=1) ** (1 / max(fine - inizio, 1)) - 1
    return pd.DataFrame({"Rendimento reale iniziale": annuo, "Patrimonio finale reale": r.patrimonio_reale[:, -1],
                         "Successo": ~r.esaurito})


def sintesi(r: Risultato) -> dict[str, float]:
    """Indicatori principali del piano, in euro di oggi."""
    cl = r.piano.cliente
    reale = r.patrimonio_reale
    col_pensione = r.colonna(min(cl.eta_pensione, cl.eta_finale))
    eta_es = r.eta_esaurimento
    in_pensione = r.eta[:-1] >= cl.eta_pensione
    spesa_reale = r.prelievi_reali[:, in_pensione]
    return {
        "Probabilità di successo": probabilita_successo(r),
        "Probabilità obiettivo": probabilita_obiettivo(r),
        "Rischio di longevità": rischio_longevita(r),
        "Patrimonio alla pensione (mediana)": float(np.median(reale[:, col_pensione])),
        "Patrimonio alla pensione (P10)": float(np.percentile(reale[:, col_pensione], 10)),
        "Patrimonio finale (mediana)": float(np.median(reale[:, -1])),
        "Patrimonio finale (P10)": float(np.percentile(reale[:, -1], 10)),
        "Patrimonio finale (P90)": float(np.percentile(reale[:, -1], 90)),
        "Età di esaurimento (mediana se esaurito)": float(np.nanmedian(eta_es)) if np.isfinite(eta_es).any()
        else float("nan"),
        "Prelievo annuo medio in pensione": float(np.median(spesa_reale.mean(axis=1))) if spesa_reale.size
        else 0.0,
        "Imposte sui prelievi (mediana)": float(np.median((r.imposte / r.indice_prezzi[:, :-1]).sum(axis=1))),
        "Inflazione media annua (mediana)": float(np.median(r.inflazione.mean(axis=1))),
        "Potere d'acquisto alla pensione": float(np.median(1 / r.indice_prezzi[:, col_pensione])),
        "Rendimento reale annuo (mediana)": float(np.median(
            ((1 + r.rendimenti) / (1 + r.inflazione)).prod(axis=1) ** (1 / r.rendimenti.shape[1]) - 1)),
    }


# --------------------------------------------------------------------------- #
# Ricerca dei valori che rendono il piano sostenibile
# --------------------------------------------------------------------------- #
def _scenari(piano: Piano, scenari: Scenari | None, n: int, seed: int | None) -> Scenari:
    return scenari if scenari is not None else genera(piano.mercato, piano.anni, n, seed)


def _risolvi(funzione, basso: float, alto: float, tolleranza: float) -> float:
    """Zero di una funzione decrescente: allarga l'intervallo verso l'alto finché cambia segno."""
    if funzione(basso) < 0:
        return float("nan")
    for _ in range(30):
        if funzione(alto) < 0:
            return float(brentq(funzione, basso, alto, xtol=tolleranza))
        basso, alto = alto, alto * 2
    return float("inf")


def prelievo_sostenibile(piano: Piano, probabilita: float = 0.90, scenari: Scenari | None = None,
                         n: int = 5_000, seed: int | None = 42) -> float:
    """Spesa annua massima in pensione (euro di oggi, pensione pubblica inclusa) con la probabilità di
    successo richiesta, prelevando un importo fisso indicizzato all'inflazione."""
    sc = _scenari(piano, scenari, n, seed)
    base = piano.con(flussi__strategia="fisso")

    def scarto(spesa: float) -> float:
        return probabilita_successo(simula(base.con(flussi__spesa_annua=spesa), sc)) - probabilita

    minimo = piano.flussi.pensione_pubblica
    return _risolvi(scarto, minimo, max(minimo * 2, minimo + 20_000), tolleranza=50.0)


def versamento_necessario(piano: Piano, probabilita: float = 0.80, criterio: str = "obiettivo",
                          scenari: Scenari | None = None, n: int = 5_000, seed: int | None = 42) -> float:
    """Versamento annuo (euro di oggi) che porta la probabilità dell'obiettivo, o del successo dell'intero piano
    con ``criterio="successo"``, al livello richiesto. Restituisce 0 se il piano lo raggiunge già senza versare."""
    if criterio not in ("obiettivo", "successo"):
        raise ValueError("criterio deve essere 'obiettivo' o 'successo'")
    if criterio == "obiettivo" and piano.obiettivo is None:
        raise ValueError("Il piano non ha un obiettivo patrimoniale")
    sc = _scenari(piano, scenari, n, seed)
    misura = probabilita_obiettivo if criterio == "obiettivo" else probabilita_successo

    def scarto(versamento: float) -> float:
        return probabilita - misura(simula(piano.con(flussi__versamento_annuo=versamento), sc))

    if scarto(0.0) <= 0:
        return 0.0
    return _risolvi(scarto, 0.0, max(piano.flussi.versamento_annuo, 5_000.0), tolleranza=50.0)


def griglia(piano: Piano, spese: Sequence[float] | None = None, quote: Sequence[float] | None = None,
            scenari: Scenari | None = None, n: int = 2_000, seed: int | None = 42) -> pd.DataFrame:
    """Probabilità di successo al variare della spesa in pensione (righe) e della quota azionaria (colonne)."""
    sc = _scenari(piano, scenari, n, seed)
    if spese is None:
        centro = piano.flussi.spesa_annua
        spese = [round(centro * f / 1_000) * 1_000 for f in (0.8, 0.875, 0.95, 1.0, 1.05, 1.125, 1.2)]
        spese = sorted(set(spese))
    if quote is None:
        quote = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    dati = {q: [probabilita_successo(simula(piano.con_quota_azionaria(q).con(flussi__spesa_annua=s), sc))
                for s in spese] for q in quote}
    return pd.DataFrame(dati, index=pd.Index(spese, name="Spesa annua"), columns=pd.Index(quote,
                                                                                         name="Quota azionaria"))


def confronta_strategie(piano: Piano, scenari: Scenari | None = None, n: int = 5_000,
                        seed: int | None = 42) -> pd.DataFrame:
    """Prelievo fisso, in percentuale del patrimonio e con guardrail a confronto sugli stessi scenari."""
    sc = _scenari(piano, scenari, n, seed)
    nomi = {"fisso": "Importo fisso indicizzato", "percentuale": f"{piano.flussi.tasso_prelievo:.1%} del patrimonio",
            "guardrail": "Fisso con guardrail"}
    prog = flussi_programmati(piano)
    in_pensione = prog["in_pensione"]
    righe = {}
    for strategia, nome in nomi.items():
        r = simula(piano.con(flussi__strategia=strategia), sc)
        # spesa sostenuta ogni anno di pensione: prelievo netto meno le uscite straordinarie, più la pensione pubblica
        spesa = (r.prelievi_reali - prog["uscite"] + prog["pensione"])[:, in_pensione]
        righe[nome.replace(".", ",")] = {
            "Probabilità di successo": probabilita_successo(r),
            "Spesa annua media": float(np.median(spesa.mean(axis=1))) if spesa.size else 0.0,
            "Spesa nell'anno peggiore (P10)": float(np.percentile(spesa.min(axis=1), 10)) if spesa.size else 0.0,
            "Patrimonio finale (mediana)": float(np.median(r.patrimonio_reale[:, -1])),
        }
    return pd.DataFrame(righe).T


def leve(piano: Piano, scenari: Scenari | None = None, n: int = 5_000, seed: int | None = 42) -> pd.DataFrame:
    """Effetto di singole modifiche al piano (spesa, risparmio, età di pensione, allocazione, strategia)."""
    sc = _scenari(piano, scenari, n, seed)
    fl, cl = piano.flussi, piano.cliente
    varianti: dict[str, Piano] = {"Piano attuale": piano}
    if fl.spesa_annua > 0:
        varianti["Spesa in pensione −10%"] = piano.con(flussi__spesa_annua=fl.spesa_annua * 0.9)
    if fl.versamento_annuo > 0:
        varianti["Versamenti +25%"] = piano.con(flussi__versamento_annuo=fl.versamento_annuo * 1.25)
    if cl.eta_pensione + 2 < cl.eta_finale:
        varianti["Pensione 2 anni più tardi"] = piano.con(cliente__eta_pensione=cl.eta_pensione + 2)
    quota = piano.quota_azionaria
    nota = ", senza life-cycle" if piano.portafoglio.pesi_pensione is not None else ""
    if quota >= 0.2:
        varianti[f"Azioni −20 punti{nota}"] = piano.con_quota_azionaria(quota - 0.2)
    if quota <= 0.8:
        varianti[f"Azioni +20 punti{nota}"] = piano.con_quota_azionaria(quota + 0.2)
    if fl.strategia != "guardrail":
        varianti["Prelievi con guardrail"] = piano.con(flussi__strategia="guardrail")
    righe = {}
    for nome, variante in varianti.items():
        r = simula(variante, sc)
        righe[nome] = {"Probabilità di successo": probabilita_successo(r),
                       "Probabilità obiettivo": probabilita_obiettivo(r),
                       "Patrimonio finale (mediana)": float(np.median(r.patrimonio_reale[:, -1]))}
    return pd.DataFrame(righe).T
