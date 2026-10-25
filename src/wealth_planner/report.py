"""Grafici interattivi (plotly) e report HTML per il cliente."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs, get_plotlyjs_version

from . import analisi
from .modello import Piano
from .scenari import Scenari, genera
from .simulazione import Risultato, flussi_programmati, simula

BLU, ARANCIO, ACQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLU_SCURO, ROSSO, GRIGIO_NEUTRO = "#1c5cab", "#e34948", "#f0efec"
TESTO, TESTO_SECONDARIO, GRIGLIA = "#222222", "#52514e", "#e6e5e1"
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


def _layout(fig: go.Figure, titolo: str, altezza: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": titolo, "x": 0, "font": {"size": 15, "color": TESTO}},
        height=altezza,
        template="plotly_white",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": TESTO_SECONDARIO},
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        separators=",.",
        hoverlabel={"bgcolor": "white", "font_color": TESTO},
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRIGLIA, zeroline=False)
    fig.update_yaxes(gridcolor=GRIGLIA, zerolinecolor="#bdbcb6")
    return fig


@dataclass
class Analisi:
    """Risultato completo dell'analisi, usato per grafici e report."""

    piano: Piano
    scenari: Scenari
    risultato: Risultato
    indicatori: dict[str, float]
    percentili: pd.DataFrame
    percentili_nominali: pd.DataFrame
    probabilita: pd.Series
    sequenza: pd.DataFrame
    griglia: pd.DataFrame
    strategie: pd.DataFrame
    leve: pd.DataFrame
    prelievo_sostenibile: float
    versamento_necessario: float
    probabilita_obiettivo_richiesta: float = 0.80
    probabilita_successo_richiesta: float = 0.90


def analizza(piano: Piano, n_simulazioni: int = 5_000, seed: int | None = 42) -> Analisi:
    """Esegue tutte le analisi del report sugli stessi scenari di mercato."""
    sc = genera(piano.mercato, piano.anni, n_simulazioni, seed)
    r = simula(piano, sc)
    if piano.obiettivo is not None:
        versamento = analisi.versamento_necessario(piano, 0.80, "obiettivo", scenari=sc)
    elif piano.cliente.eta < piano.cliente.eta_pensione:
        versamento = analisi.versamento_necessario(piano, 0.90, "successo", scenari=sc)
    else:
        versamento = float("nan")
    return Analisi(
        piano=piano,
        scenari=sc,
        risultato=r,
        indicatori=analisi.sintesi(r),
        percentili=analisi.percentili(r),
        percentili_nominali=analisi.percentili(r, reale=False),
        probabilita=analisi.probabilita_patrimonio(r),
        sequenza=analisi.rischio_sequenza(r),
        griglia=analisi.griglia(piano, scenari=sc),
        strategie=analisi.confronta_strategie(piano, sc),
        leve=analisi.leve(piano, sc),
        prelievo_sostenibile=analisi.prelievo_sostenibile(piano, 0.90, sc),
        versamento_necessario=versamento,
    )


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
def _linea_pensione(fig: go.Figure, piano: Piano) -> None:
    eta = piano.cliente.eta_pensione
    if piano.cliente.eta < eta < piano.cliente.eta_finale:
        fig.add_vline(x=eta, line={"color": TESTO_SECONDARIO, "width": 1, "dash": "dot"},
                      annotation_text="Pensione", annotation_position="top right")


def grafico_flussi(a: Analisi) -> go.Figure:
    """Versamenti e prelievi programmati per età, in euro di oggi."""
    prog = flussi_programmati(a.piano)
    eta = prog["eta"]
    fig = go.Figure()
    fig.add_bar(x=eta, y=prog["versamenti"], name="Versamenti", marker={"color": BLU},
                hovertemplate="Età %{x}<br>Versamento € %{y:,.0f}<extra></extra>")
    fig.add_bar(x=eta, y=-prog["prelievo"], name="Prelievi per la spesa in pensione", marker={"color": ARANCIO},
                hovertemplate="Età %{x}<br>Prelievo € %{customdata:,.0f}<extra></extra>", customdata=prog["prelievo"])
    if prog["uscite"].any():
        fig.add_bar(x=eta, y=-prog["uscite"], name="Spese straordinarie", marker={"color": ROSSO},
                    hovertemplate="Età %{x}<br>Spesa straordinaria € %{customdata:,.0f}<extra></extra>",
                    customdata=prog["uscite"])
    _linea_pensione(fig, a.piano)
    fig.update_layout(barmode="relative", bargap=0.15)
    fig.update_xaxes(title="Età", dtick=5)
    fig.update_yaxes(tickprefix="€ ", tickformat=",.0f")
    return _layout(fig, "Flussi programmati sul portafoglio (euro di oggi)", altezza=380)


def grafico_ventaglio(a: Analisi) -> go.Figure:
    """Ventaglio dei percentili del patrimonio reale per età."""
    p = a.percentili
    eta = p.index
    fig = go.Figure()
    for basso, alto, colore, nome in (("P5", "P95", "rgba(42,120,214,0.13)", "90% degli scenari"),
                                      ("P25", "P75", "rgba(42,120,214,0.30)", "50% degli scenari")):
        fig.add_scatter(x=eta, y=p[alto], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=eta, y=p[basso], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=colore,
                        name=nome, hoverinfo="skip")
    fig.add_scatter(x=eta, y=p["P50"], mode="lines", name="Scenario mediano", line={"color": BLU_SCURO, "width": 2.5},
                    customdata=p[["P5", "P25", "P75", "P95"]].to_numpy(),
                    hovertemplate=("Età %{x}<br>Mediana € %{y:,.0f}<br>P25–P75 € %{customdata[1]:,.0f} – "
                                   "%{customdata[2]:,.0f}<br>P5–P95 € %{customdata[0]:,.0f} – "
                                   "%{customdata[3]:,.0f}<extra></extra>"))
    ob = a.piano.obiettivo
    if ob is not None:
        fig.add_scatter(x=[ob.eta], y=[ob.importo], mode="markers", name=ob.descrizione,
                        marker={"symbol": "diamond", "size": 12, "color": ARANCIO, "line": {"color": "white",
                                                                                             "width": 1}},
                        hovertemplate=f"{ob.descrizione}<br>€ %{{y:,.0f}} a %{{x}} anni<extra></extra>")
    _linea_pensione(fig, a.piano)
    fig.update_xaxes(title="Età", dtick=5)
    fig.update_yaxes(tickprefix="€ ", tickformat=",.0f", rangemode="tozero")
    return _layout(fig, "Evoluzione del patrimonio in euro di oggi", altezza=460)


def grafico_obiettivo(a: Analisi) -> go.Figure:
    """Distribuzione del patrimonio reale all'età dell'obiettivo (o alla pensione)."""
    ob = a.piano.obiettivo
    eta = ob.eta if ob is not None else min(a.piano.cliente.eta_pensione, a.piano.cliente.eta_finale)
    valori = a.risultato.patrimonio_reale[:, a.risultato.colonna(eta)]
    mediana = float(np.median(valori))
    fig = go.Figure(go.Histogram(x=valori, xbins={"size": float(np.percentile(valori, 99)) / 40},
                                 marker={"color": BLU, "line": {"color": "white", "width": 1}},
                                 hovertemplate="€ %{x:,.0f}<br>%{y} simulazioni<extra></extra>"))
    linee = [(mediana, f"Mediana {_eur(mediana)}", {"color": TESTO, "width": 1, "dash": "dot"})]
    if ob is not None:
        linee.append((ob.importo, f"Obiettivo {_eur(ob.importo)}", {"color": ROSSO, "width": 2, "dash": "dash"}))
    linee.sort(key=lambda x: x[0])
    for i, (x, testo, stile) in enumerate(linee):
        lato = "top left" if i == 0 and len(linee) > 1 else "top right"
        fig.add_vline(x=x, line=stile, annotation_text=testo, annotation_position=lato)
    fig.update_xaxes(range=[0, float(np.percentile(valori, 99))])
    fig.update_xaxes(tickprefix="€ ", tickformat=",.0f", title=f"Patrimonio a {eta} anni (euro di oggi)")
    fig.update_yaxes(title="Numero di simulazioni")
    fig.update_layout(showlegend=False, bargap=0.02)
    return _layout(fig, f"Patrimonio a {eta} anni: distribuzione degli scenari", altezza=380)


def grafico_probabilita(a: Analisi) -> go.Figure:
    """Probabilità che il piano copra ancora le spese e probabilità di essere in vita, per età."""
    p = a.probabilita
    vita = analisi.sopravvivenza(a.piano.cliente, p.index)
    fig = go.Figure()
    fig.add_scatter(x=p.index, y=p.values, name="Il patrimonio copre ancora le spese", mode="lines",
                    line={"color": BLU, "width": 2.5}, hovertemplate="Età %{x}: %{y:.0%}<extra></extra>")
    fig.add_scatter(x=p.index, y=vita, name="Probabilità di essere in vita", mode="lines",
                    line={"color": TESTO_SECONDARIO, "width": 1.5, "dash": "dash"},
                    hovertemplate="Età %{x}: %{y:.0%}<extra></extra>")
    _linea_pensione(fig, a.piano)
    fig.update_xaxes(title="Età", dtick=5)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    return _layout(fig, "Il patrimonio dura più a lungo del cliente?", altezza=380)


def grafico_inflazione(a: Analisi) -> go.Figure:
    """Patrimonio mediano nominale e reale: la differenza è il potere d'acquisto eroso dall'inflazione."""
    nom, rea = a.percentili_nominali["P50"], a.percentili["P50"]
    fig = go.Figure()
    fig.add_scatter(x=rea.index, y=rea, name="In euro di oggi (potere d'acquisto)", mode="lines",
                    line={"color": BLU_SCURO, "width": 2.5}, hovertemplate="€ %{y:,.0f}")
    fig.add_scatter(x=nom.index, y=nom, name="In euro correnti (valore sull'estratto conto)", mode="lines",
                    line={"color": ARANCIO, "width": 2.5}, fill="tonexty", fillcolor="rgba(235,104,52,0.10)",
                    hovertemplate="€ %{y:,.0f}")
    _linea_pensione(fig, a.piano)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title="Età", dtick=5)
    fig.update_yaxes(tickprefix="€ ", tickformat=",.0f", rangemode="tozero")
    return _layout(fig, "Scenario mediano: valore nominale e potere d'acquisto", altezza=400)


def grafico_sequenza(a: Analisi) -> go.Figure:
    """Probabilità di successo per quinto di rendimento reale nei primi 10 anni di pensione."""
    q = _quinti_sequenza(a.sequenza)
    etichette = [f"{nome}<br>{_pct(lo)} … {_pct(hi)}" for nome, lo, hi in zip(q.index, q["Da"], q["A"])]
    fig = go.Figure(go.Bar(
        x=etichette, y=q["Successo"], marker={"color": [ARANCIO if v < 0.8 else BLU for v in q["Successo"]]},
        text=[_pct(v, 0) for v in q["Successo"]], textposition="outside",
        hovertemplate="%{x}<br>Successo %{y:.0%}<extra></extra>",
    ))
    fig.update_xaxes(title="Rendimento reale annuo del portafoglio nei primi 10 anni di pensione")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.12], title="Probabilità di successo")
    fig.update_layout(showlegend=False)
    return _layout(fig, "Rischio di sequenza: contano molto i primi anni di pensione", altezza=400)


def _quinti_sequenza(s: pd.DataFrame) -> pd.DataFrame:
    nomi = ["1° quinto (peggiore)", "2° quinto", "3° quinto", "4° quinto", "5° quinto (migliore)"]
    gruppi = pd.qcut(s["Rendimento reale iniziale"], 5, labels=nomi)
    return s.groupby(gruppi, observed=True).agg(Da=("Rendimento reale iniziale", "min"),
                                                A=("Rendimento reale iniziale", "max"), Successo=("Successo", "mean"))


def grafico_griglia(a: Analisi) -> go.Figure:
    """Mappa della probabilità di successo per spesa in pensione e quota azionaria."""
    g = a.griglia
    z = g.to_numpy()
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[_pct(q, 0) for q in g.columns],
            y=[_eur(s) for s in g.index],
            zmin=0.6,
            zmax=1.0,
            colorscale=[[0, ROSSO], [0.5, GRIGIO_NEUTRO], [1, BLU]],
            text=[[_pct(v, 0) for v in riga] for riga in z],
            texttemplate="%{text}",
            textfont={"size": 12, "color": TESTO},
            xgap=2,
            ygap=2,
            colorbar={"title": "Successo", "tickformat": ".0%"},
            hovertemplate="Spesa %{y}/anno<br>Azioni %{x}<br>Successo %{text}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Quota azionaria del portafoglio", side="bottom", showgrid=False)
    fig.update_yaxes(title="Spesa annua in pensione", autorange="reversed", showgrid=False)
    return _layout(fig, "Probabilità di successo per tenore di vita e allocazione", altezza=440)


GRAFICI = {
    "flussi": grafico_flussi,
    "ventaglio": grafico_ventaglio,
    "obiettivo": grafico_obiettivo,
    "probabilita": grafico_probabilita,
    "inflazione": grafico_inflazione,
    "sequenza": grafico_sequenza,
    "griglia": grafico_griglia,
}


def salva_grafici(a: Analisi, cartella: str | Path, scala: float = 2.0) -> list[Path]:
    """Esporta i grafici in PNG (richiede ``kaleido`` e un browser Chrome/Chromium)."""
    out = Path(cartella)
    out.mkdir(parents=True, exist_ok=True)
    percorsi = []
    for nome, funzione in GRAFICI.items():
        percorso = out / f"{nome}.png"
        funzione(a).write_image(percorso, width=900, scale=scala)
        percorsi.append(percorso)
    return percorsi


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float, decimali: int = 1) -> str:
    if pd.isna(x):
        return "–"
    return f"{x:.{decimali}%}".replace(".", ",")


def _eur(x: float) -> str:
    if pd.isna(x):
        return "–"
    if np.isinf(x):
        return "oltre ogni soglia"
    testo = f"{abs(x):,.0f}".replace(",", ".")
    return f"−€ {testo}" if x < -0.5 else f"€ {testo}"


def _giudizio(a: Analisi) -> tuple[str, str]:
    """Una frase di sintesi e una classe CSS per il riquadro di esito."""
    p, fl, cl = a.indicatori["Probabilità di successo"], a.piano.flussi, a.piano.cliente
    base = (f"Con le ipotesi attuali il patrimonio copre le spese fino a {cl.eta_finale} anni nel {_pct(p, 0)} "
            "degli scenari simulati.")
    sostenibile = a.prelievo_sostenibile
    if p >= 0.85:
        if sostenibile > fl.spesa_annua:
            return (f"{base} Il piano è solido: con 9 probabilità su 10 si potrebbe spendere fino a "
                    f"{_eur(sostenibile)} l'anno."), "ok"
        return f"{base} Il piano è solido.", "ok"
    leva = a.leve.drop("Piano attuale")["Probabilità di successo"].idxmax()
    testo = (f"{base} Per arrivare a 9 probabilità su 10 la spesa in pensione non dovrebbe superare "
             f"{_eur(sostenibile)} l'anno (oggi {_eur(fl.spesa_annua)}); tra le modifiche esaminate la più efficace è "
             f"«{leva.lower()}».")
    return testo, "attenzione" if p >= 0.70 else "ko"


def _testo_sequenza(s: pd.DataFrame) -> tuple[str, str]:
    q = _quinti_sequenza(s)["Successo"]
    return _pct(q.iloc[0], 0), _pct(q.iloc[-1], 0)


def build_report(
    a: Analisi,
    output: str | Path,
    client_name: str = "Cliente",
    data_note: str = "",
    offline: bool = False,
) -> Path:
    """Scrive il report HTML. Con ``offline=True`` include plotly.js nel file (circa 4 MB)."""
    piano, ind = a.piano, a.indicatori
    cl, fl, pf, mk = piano.cliente, piano.flussi, piano.portafoglio, piano.mercato
    sintesi, esito = _giudizio(a)
    config = {"displaylogo": False, "responsive": True}
    grafici = {nome: f(a).to_html(full_html=False, include_plotlyjs=False, config=config)
               for nome, f in GRAFICI.items()}

    strategie_nomi = {"fisso": "importo fisso indicizzato all'inflazione",
                      "percentuale": f"{_pct(fl.tasso_prelievo)} del patrimonio di ogni anno",
                      "guardrail": "importo fisso con guardrail (tagli e ripristini del 10%)"}
    eta_pp = cl.eta_pensione if fl.eta_pensione_pubblica is None else fl.eta_pensione_pubblica
    ipotesi = [
        ("Età attuale e pensione", f"{cl.eta} anni, pensione a {cl.eta_pensione}" if cl.eta < cl.eta_pensione
         else f"{cl.eta} anni, prelievi da subito"),
        ("Orizzonte del piano", f"fino a {cl.eta_finale} anni ({piano.anni} anni)"),
        ("Patrimonio investito oggi", _eur(fl.patrimonio_iniziale)),
        ("Versamento annuo", f"{_eur(fl.versamento_annuo)} fino alla pensione"
         + (f", +{_pct(fl.crescita_versamenti)} reale all'anno" if fl.crescita_versamenti else "")
         if fl.versamento_annuo else "nessuno"),
        ("Spesa annua in pensione", _eur(fl.spesa_annua)),
        ("Pensione pubblica netta", f"{_eur(fl.pensione_pubblica)} l'anno da {eta_pp} anni"
         if fl.pensione_pubblica else "non considerata"),
        ("Strategia di prelievo", strategie_nomi[fl.strategia]),
        ("Costi e imposte", f"costi {_pct(pf.costi, 2)}, bollo {_pct(pf.bollo, 1)}, plusvalenze "
                            f"{_pct(pf.aliquota_plusvalenze, 0)}"),
        ("Inflazione attesa", f"{_pct(mk.inflazione_media)} annuo (volatilità {_pct(mk.inflazione_volatilita)})"),
    ]
    if piano.obiettivo is not None:
        ob = piano.obiettivo
        ipotesi.insert(5, (ob.descrizione, f"{_eur(ob.importo)} a {ob.eta} anni"))
    for u in piano.uscite:
        ipotesi.append((u.descrizione, f"{_eur(u.importo)} a {u.eta} anni"))

    pesi_fine = pf.pesi_pensione or pf.pesi
    allocazione = [
        {"nome": asset.nome, "oggi": _pct(pf.pesi.get(k, 0.0), 0), "pensione": _pct(pesi_fine.get(k, 0.0), 0),
         "rendimento": _pct(asset.rendimento), "volatilita": _pct(asset.volatilita)}
        for k, asset in mk.asset.items() if pf.pesi.get(k, 0.0) > 0 or pesi_fine.get(k, 0.0) > 0
    ]

    tappe = sorted({cl.eta, *range((cl.eta // 5 + 1) * 5, cl.eta_finale, 5), cl.eta_pensione, cl.eta_finale})
    tappe = [e for e in tappe if cl.eta <= e <= cl.eta_finale]
    percentili = [{"eta": e, "valori": [_eur(a.percentili.loc[e, c]) for c in ("P5", "P25", "P50", "P75", "P95")],
                   "prob": _pct(a.probabilita.loc[e], 0), "forte": e in (cl.eta_pensione, cl.eta_finale)}
                  for e in tappe]

    strategie = [{"nome": nome, "successo": _pct(r["Probabilità di successo"], 0),
                  "media": _eur(r["Spesa annua media"]), "peggiore": _eur(r["Spesa nell'anno peggiore (P10)"]),
                  "finale": _eur(r["Patrimonio finale (mediana)"])} for nome, r in a.strategie.iterrows()]
    leve = [{"nome": nome, "successo": _pct(r["Probabilità di successo"], 0),
             "obiettivo": _pct(r["Probabilità obiettivo"], 0), "finale": _eur(r["Patrimonio finale (mediana)"]),
             "forte": nome == "Piano attuale"} for nome, r in a.leve.iterrows()]
    seq_peggiori, seq_migliori = _testo_sequenza(a.sequenza)

    eta_es = ind["Età di esaurimento (mediana se esaurito)"]
    kpi = [("Probabilità di successo", _pct(ind["Probabilità di successo"], 0))]
    if piano.obiettivo is not None:
        kpi.append((f"Probabilità obiettivo a {piano.obiettivo.eta} anni", _pct(ind["Probabilità obiettivo"], 0)))
    if cl.eta < cl.eta_pensione:
        kpi.append((f"Patrimonio mediano a {cl.eta_pensione} anni", _eur(ind["Patrimonio alla pensione (mediana)"])))
    else:
        kpi.append(("Rischio di longevità", _pct(ind["Rischio di longevità"], 0)))
    kpi += [
        ("Spesa sostenibile a prelievo fisso (9 casi su 10)", _eur(a.prelievo_sostenibile)),
    ]

    env = Environment(loader=PackageLoader("wealth_planner", "templates"), autoescape=select_autoescape(["html", "j2"]))
    html = env.get_template("report.html.j2").render(
        client=client_name,
        nome=piano.nome,
        today=date.today().strftime("%d/%m/%Y"),
        plotly_inline=get_plotlyjs() if offline else None,
        plotly_cdn=PLOTLY_CDN,
        sintesi=sintesi,
        esito=esito,
        kpi=kpi,
        ipotesi=ipotesi,
        allocazione=allocazione,
        life_cycle=pf.pesi_pensione is not None,
        percentili=percentili,
        strategie=strategie,
        leve=leve,
        obiettivo=piano.obiettivo,
        ob_importo=_eur(piano.obiettivo.importo) if piano.obiettivo else "",
        ob_prob=_pct(ind["Probabilità obiettivo"], 0),
        ob_mediana=_eur(np.median(a.risultato.patrimonio_reale[:, a.risultato.colonna(piano.obiettivo.eta)]))
        if piano.obiettivo else "",
        versamento_necessario=_eur(a.versamento_necessario),
        versamento_attuale=_eur(fl.versamento_annuo),
        accumulo=cl.eta < cl.eta_pensione,
        eta_pensione=cl.eta_pensione,
        eta_finale=cl.eta_finale,
        successo=_pct(ind["Probabilità di successo"], 0),
        longevita=_pct(ind["Rischio di longevità"], 0),
        eta_esaurimento="" if pd.isna(eta_es) else f"{eta_es:.0f}",
        finale_p10=_eur(ind["Patrimonio finale (P10)"]),
        finale_p50=_eur(ind["Patrimonio finale (mediana)"]),
        finale_p90=_eur(ind["Patrimonio finale (P90)"]),
        potere_acquisto=_eur(1_000 * ind["Potere d'acquisto alla pensione"]),
        inflazione_media=_pct(ind["Inflazione media annua (mediana)"]),
        rendimento_reale=_pct(ind["Rendimento reale annuo (mediana)"]),
        imposte=_eur(ind["Imposte sui prelievi (mediana)"]),
        seq_peggiori=seq_peggiori,
        seq_migliori=seq_migliori,
        prelievo_sostenibile=_eur(a.prelievo_sostenibile),
        spesa_attuale=_eur(fl.spesa_annua),
        n_simulazioni=f"{a.risultato.n:,}".replace(",", "."),
        regimi=mk.distribuzione == "regimi",
        prob_crisi=_pct(mk.prob_crisi, 0),
        grafici=grafici,
        data_note=data_note,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
