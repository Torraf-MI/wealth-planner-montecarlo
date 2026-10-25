"""Motore di simulazione: evoluzione anno per anno del patrimonio in ogni scenario.

Sequenza di ogni anno, per tutte le simulazioni insieme (calcolo vettoriale con numpy):

1. versamento (fino alla pensione), rivalutato con l'inflazione simulata;
2. prelievo per la spesa in pensione e per le uscite straordinarie, al lordo dell'imposta sulle plusvalenze
   realizzate (calcolata con il metodo del costo medio);
3. rendimento del portafoglio ribilanciato, al netto dei costi, e imposta di bollo sul valore di fine anno.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modello import Piano
from .scenari import Scenari, genera


@dataclass
class Risultato:
    """Traiettorie simulate di un piano. Gli importi sono nominali; le proprietà ``*_reale`` li deflazionano."""

    piano: Piano
    eta: np.ndarray  # (anni + 1,): età all'inizio di ogni anno; l'ultima colonna è l'età finale
    patrimonio: np.ndarray  # (n, anni + 1): valore a inizio anno, prima di versamenti e prelievi
    indice_prezzi: np.ndarray  # (n, anni + 1): inflazione cumulata, 1 oggi
    versamenti: np.ndarray  # (n, anni)
    prelievi: np.ndarray  # (n, anni): somme effettivamente ricevute dal cliente, al netto delle imposte
    prelievi_previsti: np.ndarray  # (n, anni): somme che sarebbero servite
    imposte: np.ndarray  # (n, anni): imposta sulle plusvalenze pagata sui prelievi
    rendimenti: np.ndarray  # (n, anni): rendimento nominale del portafoglio, al netto di costi e bollo
    inflazione: np.ndarray  # (n, anni)
    fattore_spesa: np.ndarray  # (n, anni): spesa effettiva / spesa prevista (diverso da 1 con i guardrail)

    @property
    def n(self) -> int:
        return self.patrimonio.shape[0]

    @property
    def patrimonio_reale(self) -> np.ndarray:
        return self.patrimonio / self.indice_prezzi

    @property
    def prelievi_reali(self) -> np.ndarray:
        return self.prelievi / self.indice_prezzi[:, :-1]

    @property
    def ammanchi(self) -> np.ndarray:
        """Spesa non coperta, nominale: positiva negli anni in cui il patrimonio non basta più."""
        return np.maximum(self.prelievi_previsti - self.prelievi, 0)

    @property
    def esaurito(self) -> np.ndarray:
        """True nelle simulazioni in cui, almeno un anno, il patrimonio non copre la spesa prevista."""
        return (self.ammanchi > 1.0).any(axis=1)

    @property
    def eta_esaurimento(self) -> np.ndarray:
        """Età del primo anno con spesa non coperta; NaN se il patrimonio basta fino alla fine."""
        mancanza = self.ammanchi > 1.0
        primo = mancanza.argmax(axis=1)
        return np.where(mancanza.any(axis=1), self.eta[primo], np.nan).astype(float)

    def colonna(self, eta: int) -> int:
        """Indice della colonna di ``patrimonio`` corrispondente all'età indicata."""
        if not self.eta[0] <= eta <= self.eta[-1]:
            raise ValueError(f"Età {eta} fuori dall'orizzonte del piano ({self.eta[0]}-{self.eta[-1]})")
        return int(eta - self.eta[0])


def pesi_annui(piano: Piano) -> np.ndarray:
    """Allocazione di ogni anno (anni × classi): interpolazione lineare verso i pesi alla pensione, poi costante."""
    chiavi = piano.mercato.chiavi
    inizio = np.array([piano.portafoglio.pesi.get(k, 0.0) for k in chiavi])
    if piano.portafoglio.pesi_pensione is None:
        return np.tile(inizio, (piano.anni, 1))
    fine = np.array([piano.portafoglio.pesi_pensione.get(k, 0.0) for k in chiavi])
    durata = max(piano.cliente.eta_pensione - piano.cliente.eta, 1)
    avanzamento = np.clip(np.arange(piano.anni) / durata, 0, 1)[:, None]
    return inizio + (fine - inizio) * avanzamento


def flussi_programmati(piano: Piano) -> dict[str, np.ndarray]:
    """Versamenti, pensione pubblica, prelievi per la spesa e uscite di ogni anno, in euro di oggi."""
    cl, fl = piano.cliente, piano.flussi
    eta = cl.eta + np.arange(piano.anni)
    in_pensione = eta >= cl.eta_pensione
    eta_pp = cl.eta_pensione if fl.eta_pensione_pubblica is None else fl.eta_pensione_pubblica
    versamenti = np.where(~in_pensione, fl.versamento_annuo * (1 + fl.crescita_versamenti) ** np.arange(piano.anni), 0)
    pensione = np.where(eta >= eta_pp, fl.pensione_pubblica, 0.0)
    spesa = np.where(in_pensione, fl.spesa_annua, 0.0)
    uscite = np.zeros(piano.anni)
    for u in piano.uscite:
        if cl.eta <= u.eta < cl.eta_finale:
            uscite[u.eta - cl.eta] += u.importo
    return {"eta": eta, "in_pensione": in_pensione, "versamenti": versamenti, "pensione": pensione,
            "spesa": spesa, "prelievo": np.maximum(spesa - pensione, 0), "uscite": uscite}


def simula(piano: Piano, scenari: Scenari | None = None, n: int = 5_000, seed: int | None = 42) -> Risultato:
    """Simula il piano su ``scenari`` (generati con ``n`` e ``seed`` se non forniti)."""
    T = piano.anni
    if scenari is None:
        scenari = genera(piano.mercato, T, n, seed)
    if scenari.anni < T or scenari.asset != piano.mercato.chiavi:
        raise ValueError("Gli scenari non coprono l'orizzonte o le classi di attività del piano")
    n = scenari.n
    fl, pf = piano.flussi, piano.portafoglio
    prog = flussi_programmati(piano)

    lordo = np.einsum("ntk,tk->nt", scenari.rendimenti[:, :T], pesi_annui(piano))
    rendimenti = (1 + lordo - pf.costi) * (1 - pf.bollo) - 1
    inflazione = scenari.inflazione[:, :T]
    indice = np.ones((n, T + 1))
    indice[:, 1:] = np.cumprod(1 + inflazione, axis=1)

    patrimonio = np.empty((n, T + 1))
    versamenti, prelievi, previsti, imposte, fattori = (np.zeros((n, T)) for _ in range(5))
    valore = np.full(n, float(fl.patrimonio_iniziale))
    carico = np.full(n, float(fl.patrimonio_iniziale if fl.costo_fiscale_iniziale is None
                              else fl.costo_fiscale_iniziale))
    fattore = np.ones(n)  # spesa effettiva / prevista (guardrail)
    tasso_iniziale: np.ndarray | None = None

    for t in range(T):
        patrimonio[:, t] = valore
        prezzi = indice[:, t]

        versato = prog["versamenti"][t] * prezzi
        valore = valore + versato
        carico = carico + versato
        versamenti[:, t] = versato

        if not prog["in_pensione"][t]:
            richiesto = np.zeros(n)
        elif fl.strategia == "percentuale":
            richiesto = fl.tasso_prelievo * valore
        elif fl.strategia == "guardrail":
            base = prog["prelievo"][t] * prezzi
            if tasso_iniziale is None:
                tasso_iniziale = np.divide(base, valore, out=np.full(n, np.inf), where=valore > 0)
            else:
                tasso = np.divide(base * fattore, valore, out=np.full(n, np.inf), where=valore > 0)
                taglio = tasso > tasso_iniziale * (1 + fl.soglia_guardrail)
                ripristino = tasso < tasso_iniziale * (1 - fl.soglia_guardrail)
                fattore = np.where(taglio, fattore * (1 - fl.correzione_guardrail), fattore)
                fattore = np.where(ripristino, fattore * (1 + fl.correzione_guardrail), fattore)
                fattore = np.clip(fattore, 1 - fl.taglio_massimo, 1.0)
            richiesto = base * fattore
        else:
            richiesto = np.full(n, prog["prelievo"][t]) * prezzi
        fattori[:, t] = fattore
        richiesto = richiesto + prog["uscite"][t] * prezzi

        # Prelievo lordo che, dopo l'imposta sulla quota di plusvalenza, lascia al cliente la somma richiesta
        quota_plus = np.clip(np.divide(valore - carico, valore, out=np.zeros(n), where=valore > 0), 0, 1)
        aliquota_eff = pf.aliquota_plusvalenze * quota_plus
        lordo_prelievo = np.minimum(richiesto / (1 - aliquota_eff), valore)
        netto = lordo_prelievo * (1 - aliquota_eff)
        carico = carico * (1 - np.divide(lordo_prelievo, valore, out=np.zeros(n), where=valore > 0))
        valore = valore - lordo_prelievo
        previsti[:, t], prelievi[:, t], imposte[:, t] = richiesto, netto, lordo_prelievo - netto

        valore = np.maximum(valore * (1 + rendimenti[:, t]), 0)

    patrimonio[:, T] = valore
    return Risultato(piano=piano, eta=piano.cliente.eta + np.arange(T + 1), patrimonio=patrimonio,
                     indice_prezzi=indice, versamenti=versamenti, prelievi=prelievi, prelievi_previsti=previsti,
                     imposte=imposte, rendimenti=rendimenti, inflazione=inflazione, fattore_spesa=fattori)
