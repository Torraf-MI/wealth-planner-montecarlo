"""Ipotesi del piano: cliente, flussi, portafoglio, mercato, obiettivo e uscite straordinarie.

Tutti gli importi sono espressi in **euro di oggi** (valori reali): il motore di simulazione li rivaluta
con l'inflazione simulata anno per anno.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class AssetClass:
    """Classe di attività con rendimento atteso e volatilità annui (nominali, lordi di costi e imposte)."""

    nome: str
    rendimento: float  # media aritmetica del rendimento annuo
    volatilita: float  # deviazione standard del rendimento annuo
    azionario: bool = False


def _asset_standard() -> dict[str, AssetClass]:
    return {
        "azioni": AssetClass("Azioni globali", rendimento=0.070, volatilita=0.160, azionario=True),
        "obbligazioni": AssetClass("Obbligazioni area euro", rendimento=0.030, volatilita=0.055),
        "liquidita": AssetClass("Liquidità e monetario", rendimento=0.022, volatilita=0.010),
    }


CORRELAZIONI_STANDARD = (
    (1.00, 0.15, 0.00),
    (0.15, 1.00, 0.25),
    (0.00, 0.25, 1.00),
)


@dataclass(frozen=True)
class Mercato:
    """Ipotesi di mercato: classi di attività, correlazioni e inflazione.

    Con ``distribuzione="regimi"`` ogni anno può essere, con probabilità ``prob_crisi``, un anno turbolento in cui
    la volatilità di tutte le attività è moltiplicata per ``moltiplicatore_crisi``. La volatilità complessiva resta
    quella indicata, ma le perdite estreme diventano più frequenti che con la distribuzione normale (code grasse).
    """

    asset: dict[str, AssetClass] = field(default_factory=_asset_standard)
    correlazioni: tuple[tuple[float, ...], ...] = CORRELAZIONI_STANDARD
    inflazione_media: float = 0.020
    inflazione_volatilita: float = 0.012
    inflazione_persistenza: float = 0.6  # autocorrelazione annua (AR(1)): l'inflazione alta tende a durare
    distribuzione: Literal["normale", "regimi"] = "regimi"
    prob_crisi: float = 0.10
    moltiplicatore_crisi: float = 2.0

    def __post_init__(self) -> None:
        k = len(self.asset)
        corr = np.asarray(self.correlazioni, dtype=float)
        if corr.shape != (k, k):
            raise ValueError(f"La matrice delle correlazioni deve essere {k}×{k}, una riga per classe di attività")
        if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1):
            raise ValueError("La matrice delle correlazioni deve essere simmetrica con 1 sulla diagonale")
        if np.linalg.eigvalsh(corr).min() < -1e-10:
            raise ValueError("La matrice delle correlazioni non è definita positiva")
        if self.distribuzione not in ("normale", "regimi"):
            raise ValueError("distribuzione deve essere 'normale' o 'regimi'")

    @property
    def chiavi(self) -> tuple[str, ...]:
        return tuple(self.asset)

    @property
    def azionarie(self) -> tuple[str, ...]:
        return tuple(k for k, a in self.asset.items() if a.azionario)


@dataclass(frozen=True)
class Cliente:
    """Età e orizzonte del piano. La longevità segue una legge di Gompertz (età modale e dispersione)."""

    eta: int
    eta_pensione: int = 67
    eta_finale: int = 95  # fino a quando il patrimonio deve bastare
    eta_modale_decesso: float = 90.0  # età di decesso più frequente
    dispersione_decesso: float = 9.0

    def __post_init__(self) -> None:
        if not self.eta < self.eta_finale:
            raise ValueError("L'età finale deve essere successiva all'età attuale")


@dataclass(frozen=True)
class Flussi:
    """Patrimonio di partenza, versamenti durante la vita lavorativa e spesa in pensione (euro di oggi)."""

    patrimonio_iniziale: float
    versamento_annuo: float = 0.0  # risparmio investito ogni anno fino alla pensione
    crescita_versamenti: float = 0.0  # crescita reale annua dei versamenti (carriera)
    spesa_annua: float = 0.0  # tenore di vita in pensione, comprese le entrate da pensione pubblica
    pensione_pubblica: float = 0.0  # pensione netta annua, indicizzata all'inflazione
    eta_pensione_pubblica: int | None = None  # se assente coincide con l'età di pensione
    costo_fiscale_iniziale: float | None = None  # prezzo di carico; se assente, nessuna plusvalenza latente
    strategia: Literal["fisso", "percentuale", "guardrail"] = "fisso"
    tasso_prelievo: float = 0.04  # strategia "percentuale": quota del patrimonio prelevata ogni anno
    soglia_guardrail: float = 0.20  # strategia "guardrail": scostamento tollerato dal tasso di prelievo iniziale
    correzione_guardrail: float = 0.10  # taglio o ripristino della spesa quando si supera la soglia
    taglio_massimo: float = 0.30  # la spesa non scende mai sotto il 70% di quella prevista

    def __post_init__(self) -> None:
        if self.strategia not in ("fisso", "percentuale", "guardrail"):
            raise ValueError("strategia deve essere 'fisso', 'percentuale' o 'guardrail'")


@dataclass(frozen=True)
class Portafoglio:
    """Allocazione, costi e fiscalità del portafoglio.

    Se ``pesi_pensione`` è indicato, l'allocazione passa gradualmente da ``pesi`` a ``pesi_pensione`` tra oggi e
    l'età di pensione (percorso *life-cycle*) e poi resta costante. Il portafoglio è ribilanciato ogni anno.
    """

    pesi: dict[str, float] = field(default_factory=lambda: {"azioni": 0.60, "obbligazioni": 0.35, "liquidita": 0.05})
    pesi_pensione: dict[str, float] | None = None
    costi: float = 0.004  # costi annui totali (TER dei fondi o ETF ed eventuale consulenza)
    bollo: float = 0.002  # imposta di bollo sui prodotti finanziari
    aliquota_plusvalenze: float = 0.26  # imposta sulle plusvalenze realizzate con i prelievi


@dataclass(frozen=True)
class Obiettivo:
    """Patrimonio reale da raggiungere a una certa età (es. capitale alla pensione)."""

    importo: float
    eta: int
    descrizione: str = "Obiettivo patrimoniale"


@dataclass(frozen=True)
class Uscita:
    """Spesa straordinaria prelevata dal portafoglio in un anno preciso (es. università dei figli, casa)."""

    eta: int
    importo: float
    descrizione: str = "Spesa straordinaria"


def pesi_con_quota_azionaria(pesi: dict[str, float], mercato: Mercato, quota: float) -> dict[str, float]:
    """Riscala i pesi in modo che la componente azionaria valga ``quota``, mantenendo le proporzioni interne."""
    if not 0 <= quota <= 1:
        raise ValueError("La quota azionaria deve essere tra 0 e 1")
    azionarie = set(mercato.azionarie)
    gruppi = {True: [k for k in mercato.chiavi if k in azionarie], False: [k for k in mercato.chiavi
                                                                          if k not in azionarie]}
    nuovi: dict[str, float] = {}
    for azionario, chiavi in gruppi.items():
        target = quota if azionario else 1 - quota
        if not chiavi:
            continue
        totale = sum(pesi.get(k, 0.0) for k in chiavi)
        for k in chiavi:
            base = pesi.get(k, 0.0) / totale if totale > 0 else 1 / len(chiavi)
            nuovi[k] = target * base
    return nuovi


@dataclass(frozen=True)
class Piano:
    """Insieme completo delle ipotesi di un piano patrimoniale."""

    cliente: Cliente
    flussi: Flussi
    portafoglio: Portafoglio = field(default_factory=Portafoglio)
    mercato: Mercato = field(default_factory=Mercato)
    obiettivo: Obiettivo | None = None
    uscite: tuple[Uscita, ...] = ()
    nome: str = "Piano patrimoniale"

    def __post_init__(self) -> None:
        for nome, pesi in (("pesi", self.portafoglio.pesi), ("pesi_pensione", self.portafoglio.pesi_pensione)):
            if pesi is None:
                continue
            sconosciute = set(pesi) - set(self.mercato.chiavi)
            if sconosciute:
                raise ValueError(f"Classi di attività non definite nel mercato: {', '.join(sorted(sconosciute))}")
            if abs(sum(pesi.values()) - 1) > 1e-6:
                raise ValueError(f"I {nome} del portafoglio devono sommare a 1 (ora {sum(pesi.values()):.3f})")

    @property
    def anni(self) -> int:
        return self.cliente.eta_finale - self.cliente.eta

    @property
    def quota_azionaria(self) -> float:
        return sum(self.portafoglio.pesi.get(k, 0.0) for k in self.mercato.azionarie)

    def con(self, **modifiche: Any) -> Piano:
        """Copia con alcune ipotesi cambiate, indicate come ``sezione__campo``.

        Esempio: ``piano.con(flussi__spesa_annua=36_000, cliente__eta_pensione=65)``.
        """
        sezioni: dict[str, dict[str, Any]] = {}
        semplici: dict[str, Any] = {}
        for chiave, valore in modifiche.items():
            if "__" in chiave:
                sezione, campo = chiave.split("__", 1)
                sezioni.setdefault(sezione, {})[campo] = valore
            else:
                semplici[chiave] = valore
        for sezione, valori in sezioni.items():
            semplici[sezione] = replace(getattr(self, sezione), **valori)
        return replace(self, **semplici)

    def con_quota_azionaria(self, quota: float) -> Piano:
        """Copia con allocazione costante e quota azionaria ``quota`` (senza percorso life-cycle)."""
        pesi = pesi_con_quota_azionaria(self.portafoglio.pesi, self.mercato, quota)
        return self.con(portafoglio__pesi=pesi, portafoglio__pesi_pensione=None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Piano:
        nomi_validi = {f.name for f in fields(cls)}
        sconosciute = set(d) - nomi_validi
        if sconosciute:
            raise ValueError(f"Sezioni non riconosciute: {', '.join(sorted(sconosciute))}")
        kwargs: dict[str, Any] = {"nome": d.get("nome", "Piano patrimoniale"),
                                  "cliente": Cliente(**d["cliente"]), "flussi": Flussi(**d["flussi"])}
        if "portafoglio" in d:
            kwargs["portafoglio"] = Portafoglio(**d["portafoglio"])
        if "mercato" in d:
            kwargs["mercato"] = _mercato_da_dict(d["mercato"])
        if "obiettivo" in d:
            kwargs["obiettivo"] = Obiettivo(**d["obiettivo"])
        if "uscite" in d:
            kwargs["uscite"] = tuple(Uscita(**u) for u in d["uscite"])
        return cls(**kwargs)


def _mercato_da_dict(d: dict[str, Any]) -> Mercato:
    """Le classi di attività indicate sostituiscono o integrano quelle standard, campo per campo."""
    d = dict(d)
    asset = _asset_standard()
    for chiave, valori in d.pop("asset", {}).items():
        if chiave in asset:
            asset[chiave] = replace(asset[chiave], **valori)
        else:
            asset[chiave] = AssetClass(**valori)
    if "correlazioni" in d:
        d["correlazioni"] = tuple(tuple(r) for r in d["correlazioni"])
    return Mercato(asset=asset, **d)


def load_toml(path: str | Path) -> Piano:
    """Legge le ipotesi da un file TOML con sezioni [cliente], [flussi], [portafoglio], [mercato], [obiettivo]
    e una tabella [[uscite]] per ogni spesa straordinaria."""
    with open(path, "rb") as f:
        return Piano.from_dict(tomllib.load(f))


def esempio_rossi() -> Piano:
    """Professionista di 45 anni che accumula per la pensione a 67, con l'università dei figli a metà strada."""
    return Piano(
        nome="Famiglia Rossi, piano pensione",
        cliente=Cliente(eta=45, eta_pensione=67, eta_finale=95),
        flussi=Flussi(patrimonio_iniziale=180_000, versamento_annuo=14_000, crescita_versamenti=0.01,
                      spesa_annua=40_000, pensione_pubblica=21_000),
        portafoglio=Portafoglio(pesi={"azioni": 0.70, "obbligazioni": 0.25, "liquidita": 0.05},
                                pesi_pensione={"azioni": 0.45, "obbligazioni": 0.45, "liquidita": 0.10}),
        obiettivo=Obiettivo(importo=550_000, eta=67, descrizione="Capitale alla pensione"),
        uscite=(Uscita(eta=53, importo=30_000, descrizione="Università del primo figlio"),
                Uscita(eta=56, importo=30_000, descrizione="Università del secondo figlio")),
    )
