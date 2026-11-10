"""EHCVM 01 — Table ménage : socle welfare + logement + actifs + GPS.

Entrée  : archive ZIP EHCVM 2021-2022 (config.yaml), lue sans extraction.
Sortie  : data/interim/ehcvm/ehcvm_households.csv, une ligne par ménage.

Règles (issues de l'audit reports/ehcvm/AUDIT_BASES_EHCVM_2021_2022.md) :
- socle = ehcvm_welfare (12 965 ménages) ; clé (grappe, menage, vague) ;
- chaque fusion est validée un-à-un et doit être complète ;
- actifs s12 mis en colonnes : asset_<code> = 1 (code 1), 0 (code 2), NaN sinon ;
  une valeur manquante n'est jamais assimilée à une absence de bien ;
- seules les variables nécessaires sont conservées.
"""
from pathlib import Path
import sys
from zipfile import ZipFile

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.data.ehcvm.audit_archive import KEYS, check_archive, default_archive
from src.utils.config import load_config
from src.utils.io import check_full_merge, write_table
from src.utils.paths import EHCVM_INTERIM_DIR, ensure_layout

OUTPUT_FILE = EHCVM_INTERIM_DIR / "ehcvm_households.csv"
WELFARE_COLUMNS = ["hhweight", "hhsize", "milieu", "region", "pcexp"]
GPS_COLUMNS = {"GPS__Latitude": "latitude", "GPS__Longitude": "longitude"}


def read_module(archive, members, name, columns, encoding):
    with archive.open(members[f"{name}_civ2021.csv"]) as handle:
        frame = pd.read_csv(handle, encoding=encoding, usecols=columns, low_memory=False)
    return frame


def assets_wide(assets):
    """Une colonne par article ; code 1 → 1, code 2 → 0, autre → NaN."""
    if assets.duplicated(KEYS + ["s12q01"]).any():
        raise ValueError("Articles dupliqués dans s12 : pivot impossible.")
    assets = assets.assign(owned=assets["s12q02"].map({1: 1.0, 2: 0.0}))
    wide = assets.pivot(index=KEYS, columns="s12q01", values="owned")
    wide.columns = [f"asset_{int(code):02d}" for code in wide.columns]
    return wide.reset_index()


def main():
    ensure_layout("ehcvm")
    config = load_config()["ehcvm"]
    housing_vars = config["wealth_index"]["housing_variables"] + ["s11q02"]
    archive_path = default_archive()
    members = check_archive(archive_path)
    encoding = config["encoding"]
    with ZipFile(archive_path) as archive:
        welfare = read_module(archive, members, "ehcvm_welfare", KEYS + WELFARE_COLUMNS, encoding)
        housing = read_module(archive, members, "s11_me", KEYS + housing_vars, encoding)
        assets = read_module(archive, members, "s12_me", KEYS + ["s12q01", "s12q02"], encoding)
        gps = read_module(archive, members, "s00_me", KEYS + list(GPS_COLUMNS), encoding)

    if welfare.duplicated(KEYS).any() or welfare[KEYS].isna().any().any():
        raise ValueError("Clé ménage invalide dans welfare.")
    table = welfare
    for label, frame in [("logement", housing), ("actifs", assets_wide(assets)),
                         ("GPS", gps.rename(columns=GPS_COLUMNS))]:
        # how="left" : les 728 clés de s00 hors welfare ne sont pas intégrées ;
        # tout ménage welfare absent d'un module (left_only) arrête l'étape.
        table = check_full_merge(
            table.merge(frame, on=KEYS, how="left", validate="one_to_one", indicator=True),
            label=f"welfare × {label}")
    n_asset_missing = int(table.filter(like="asset_").isna().sum().sum())
    print(f"Ménages : {len(table)} | grappes : {table['grappe'].nunique()} "
          f"| articles : {table.filter(like='asset_').shape[1]} | codes d'actifs invalides : {n_asset_missing}")
    write_table(table, OUTPUT_FILE)


if __name__ == "__main__":
    main()
