"""Uso del pacchetto come libreria: simulazione, indicatori, leve e report."""

from wealth_planner import (
    esempio_rossi,
    genera,
    prelievo_sostenibile,
    probabilita_obiettivo,
    probabilita_successo,
    simula,
    sintesi,
    versamento_necessario,
)
from wealth_planner.analisi import confronta_strategie, leve, percentili
from wealth_planner.report import analizza, build_report

piano = esempio_rossi()

# Gli stessi scenari di mercato per tutti i confronti: le differenze dipendono solo dalle scelte del piano
scenari = genera(piano.mercato, piano.anni, n=5_000, seed=42)
r = simula(piano, scenari)

print(f"Probabilità di successo fino a 95 anni: {probabilita_successo(r):.1%}")
print(f"Probabilità di avere 550.000 € a 67 anni: {probabilita_obiettivo(r):.1%}")
print(percentili(r).loc[[45, 55, 67, 80, 95]].round(-3))

for nome, valore in sintesi(r).items():
    print(f"  {nome:<45} {valore:,.3f}")

# Quanto si può spendere con 9 probabilità su 10? Quanto bisogna risparmiare per l'obiettivo?
print(f"Spesa sostenibile al 90%: € {prelievo_sostenibile(piano, 0.90, scenari):,.0f}")
print(f"Versamento per l'obiettivo all'80%: € {versamento_necessario(piano, 0.80, scenari=scenari):,.0f}")

# Varianti: pensione a 69 anni, spesa più bassa, allocazione più prudente
print(probabilita_successo(simula(piano.con(cliente__eta_pensione=69), scenari)))
print(probabilita_successo(simula(piano.con(flussi__spesa_annua=36_000), scenari)))
print(probabilita_successo(simula(piano.con_quota_azionaria(0.4), scenari)))

print(confronta_strategie(piano, scenari))
print(leve(piano, scenari))

build_report(analizza(piano), "output/report.html", client_name="Mario Rossi")
