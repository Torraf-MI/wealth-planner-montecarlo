from pathlib import Path

import pytest

from wealth_planner import Piano, esempio_rossi, load_toml
from wealth_planner.cli import main
from wealth_planner.report import GRAFICI, analizza, build_report

ESEMPI = Path(__file__).parents[1] / "examples"


def test_toml_equivale_all_esempio():
    assert load_toml(ESEMPI / "famiglia_rossi.toml") == esempio_rossi()


def test_toml_decumulo_con_mercato_personalizzato():
    p = load_toml(ESEMPI / "pensionato_decumulo.toml")
    assert p.flussi.strategia == "guardrail"
    assert p.mercato.asset["obbligazioni"].rendimento == pytest.approx(0.032)
    assert p.mercato.asset["azioni"].rendimento == pytest.approx(0.07)  # invariato


def test_sezioni_e_pesi_non_validi():
    d = {"cliente": {"eta": 40}, "flussi": {"patrimonio_iniziale": 1}}
    with pytest.raises(ValueError, match="Sezioni"):
        Piano.from_dict({**d, "immobili": {}})
    with pytest.raises(ValueError, match="sommare"):
        Piano.from_dict({**d, "portafoglio": {"pesi": {"azioni": 0.5}}})


@pytest.fixture(scope="module")
def analisi_demo():
    return analizza(esempio_rossi(), n_simulazioni=500)


def test_grafici(analisi_demo):
    for funzione in GRAFICI.values():
        assert funzione(analisi_demo).data


def test_report(analisi_demo, tmp_path):
    path = build_report(analisi_demo, tmp_path / "report.html", client_name="Mario Rossi")
    html = path.read_text(encoding="utf-8")
    assert "Mario Rossi" in html and "Probabilità di successo" in html and "Capitale alla pensione" in html


def test_cli(tmp_path, capsys):
    out = tmp_path / "r.html"
    assert main(["--config", str(ESEMPI / "pensionato_decumulo.toml"), "--simulazioni", "300",
                 "--output", str(out)]) == 0
    assert out.exists()
    assert "Probabilità di successo" in capsys.readouterr().out
