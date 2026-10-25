"""Riga di comando: simula un piano patrimoniale e genera il report per il cliente.

Esempi::

    wealth-planner --demo --cliente "Mario Rossi"
    wealth-planner --config examples/pensionato_decumulo.toml --offline
    wealth-planner --demo --spesa 36000 --eta-pensione 69 --strategia guardrail
"""

from __future__ import annotations

import argparse
import sys

from .modello import esempio_rossi, load_toml
from .report import _eur, _pct, analizza, build_report, salva_grafici


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="wealth-planner", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="File TOML con le ipotesi del piano")
    src.add_argument("--demo", action="store_true", help="Usa l'esempio di una famiglia che accumula per la pensione")
    p.add_argument("--patrimonio", type=float, help="Sostituisce il patrimonio investito oggi")
    p.add_argument("--versamento", type=float, help="Sostituisce il versamento annuo")
    p.add_argument("--spesa", type=float, help="Sostituisce la spesa annua in pensione (euro di oggi)")
    p.add_argument("--eta-pensione", type=int, help="Sostituisce l'età di pensione")
    p.add_argument("--strategia", choices=["fisso", "percentuale", "guardrail"], help="Strategia di prelievo")
    p.add_argument("--azioni", type=float, help="Quota azionaria costante (0-1), senza percorso life-cycle")
    p.add_argument("--simulazioni", type=int, default=5_000, help="Scenari Monte Carlo (default 5000)")
    p.add_argument("--seed", type=int, default=42, help="Seme del generatore casuale, per risultati ripetibili")
    p.add_argument("--cliente", default="Cliente", help="Nome del cliente nel report")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--offline", action="store_true", help="Include plotly.js nel report (file autonomo, ~4 MB)")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG (richiede kaleido)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    piano = load_toml(args.config) if args.config else esempio_rossi()
    modifiche = {}
    if args.patrimonio is not None:
        modifiche["flussi__patrimonio_iniziale"] = args.patrimonio
    if args.versamento is not None:
        modifiche["flussi__versamento_annuo"] = args.versamento
    if args.spesa is not None:
        modifiche["flussi__spesa_annua"] = args.spesa
    if args.eta_pensione is not None:
        modifiche["cliente__eta_pensione"] = args.eta_pensione
    if args.strategia:
        modifiche["flussi__strategia"] = args.strategia
    if modifiche:
        piano = piano.con(**modifiche)
    if args.azioni is not None:
        piano = piano.con_quota_azionaria(args.azioni)

    a = analizza(piano, n_simulazioni=args.simulazioni, seed=args.seed)
    note = "Esempio dimostrativo con valori plausibili, non riferiti a persone reali." if args.demo else ""
    path = build_report(a, args.output, client_name=args.cliente, data_note=note, offline=args.offline)
    if args.grafici:
        salva_grafici(a, args.grafici)

    ind = a.indicatori
    print(piano.nome)
    righe = [("Probabilità di successo", _pct(ind["Probabilità di successo"]))]
    if piano.obiettivo is not None:
        righe.append(("Probabilità obiettivo", _pct(ind["Probabilità obiettivo"])))
    righe += [
        ("Rischio di longevità", _pct(ind["Rischio di longevità"])),
    ]
    if piano.cliente.eta < piano.cliente.eta_pensione:
        righe.append(("Patrimonio alla pensione",
                      _eur(ind["Patrimonio alla pensione (mediana)"]) + " (mediana, euro di oggi)"))
    righe += [
        ("Patrimonio finale", _eur(ind["Patrimonio finale (mediana)"]) + " (mediana, euro di oggi)"),
        ("Spesa sostenibile 90%", _eur(a.prelievo_sostenibile) + " l'anno"),
    ]
    for nome, valore in righe:
        print(f"  {nome:<26} {valore}")
    print(f"\nReport salvato in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
