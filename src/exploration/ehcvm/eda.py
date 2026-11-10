"""Analyse exploratoire EHCVM 2021–2022, avant ACP et modélisation."""
from pathlib import Path
import argparse
import sys
from zipfile import ZipFile

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import EHCVM_OUTPUTS_DIR
from src.data.ehcvm.audit_archive import default_archive, check_archive, KEYS
from src.utils.statistics import weighted_summary, category_frequencies, missing_profile, weighted_correlation
from src.utils.reporting import prepare_output, save_table, save_figure, render_report, write_manifest

HOUSING = {
    's11q01': 'Type de logement', 's11q02': 'Nombre de pièces',
    's11q18': 'Matériaux des murs', 's11q19': 'Matériau du toit',
    's11q20': 'Revêtement du sol', 's11q21': 'Raccordement à l’eau',
    's11q26a': 'Eau de boisson, saison sèche', 's11q26b': 'Eau de boisson, saison pluvieuse',
    's11q33': 'Raccordement électrique (plusieurs modalités)',
    's11q37': 'Source d’éclairage', 's11q54': 'Type de sanitaire',
}
SOURCES = [
 ('Dictionnaire des modules EHCVM', 'https://microdata.worldbank.org/catalog/6273/data-dictionary'),
 ('Logement : libellés des questions', 'https://microdata.worldbank.org/catalog/6273/data-dictionary/F22?file_name=s11_me_civ2021'),
 ('Actifs : codes des articles et questions', 'https://microdata.worldbank.org/index.php/catalog/6273/data-dictionary/F23'),
 ('Bien-être : définition des champs', 'https://microdata.worldbank.org/catalog/6273/data-dictionary/F58?file_name=ehcvm_welfare_civ2021'),
]


def analysis(archive_path, output):
    members = check_archive(archive_path)
    data = {}
    with ZipFile(archive_path) as archive:
        for name in ['ehcvm_welfare', 's11_me', 's12_me', 's00_me', 'ehcvm_menage']:
            with archive.open(members[name+'_civ2021.csv']) as handle:
                data[name] = pd.read_csv(handle, encoding='cp1252', low_memory=False)
    welfare, housing, assets = (data[n] for n in ['ehcvm_welfare', 's11_me', 's12_me'])
    for name, frame in [('welfare', welfare), ('logement', housing)]:
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f'Clés non valides dans {name}')
    if assets.duplicated(KEYS+['s12q01']).any():
        raise ValueError('Articles dupliqués : résoudre avant pivot')
    hh = welfare.merge(housing[KEYS+list(HOUSING)], on=KEYS, how='left', validate='one_to_one', indicator=True)
    if not hh._merge.eq('both').all():
        raise ValueError('Ménages welfare sans logement ; examiner les exclusions')
    hh = hh.drop(columns='_merge')
    pivot = assets.pivot(index=KEYS, columns='s12q01', values='s12q02').add_prefix('article_').reset_index()
    hh = hh.merge(pivot, on=KEYS, how='left', validate='one_to_one')
    asset_cols = [c for c in hh if c.startswith('article_')]
    weights = hh.hhweight.to_numpy(float)
    weighted_summary(hh.pcexp, weights)  # contrôle des poids
    output = prepare_output(output)
    # Codes 1 et 2 gardés explicitement : aucun libellé d'article inventé.
    binary = hh[asset_cols].apply(lambda col: col.map({1: 1., 2: 0.}))
    invalid_asset_codes = int((hh[asset_cols].notna() & ~hh[asset_cols].isin([1, 2])).sum().sum())
    inventory = pd.DataFrame([{'module': name, 'rows': len(frame), 'columns': len(frame.columns),
                               'missing_cells': int(frame.isna().sum().sum())} for name, frame in data.items()])
    save_table(inventory, output, 'inventaire_modules')
    raw_missing = pd.concat([pd.DataFrame({'module': name, 'variable': frame.columns,
                                           'n_missing': frame.isna().sum().to_numpy(),
                                           'missing_rate': frame.isna().mean().to_numpy()}) for name, frame in data.items()], ignore_index=True)
    save_table(raw_missing, output, 'manquants_modules_bruts')
    missing = missing_profile(hh, list(HOUSING)+['hhweight', 'hhsize', 'pcexp']+asset_cols, weights)
    save_table(missing, output, 'manquants_variables_candidates')
    dictionary = pd.DataFrame([{'variable': name, 'question': label, 'type': 'quantitative' if name=='s11q02' else 'categorielle',
                                'category_labels_status': 'modalités conservées en codes ; dictionnaire complet à valider'} for name, label in HOUSING.items()])
    save_table(dictionary, output, 'dictionnaire_candidats')
    group_defs = [('ensemble', 'tous', hh.index)]
    for group_col in ['milieu', 'region', 'vague']:
        group_defs += [(group_col, str(code), rows.index) for code, rows in hh.groupby(group_col, dropna=False)]
    summaries, compositions, frequencies, asset_rates = [], [], [], []
    for group_col, group, index in group_defs:
        frame = hh.loc[index]; w = frame.hhweight.to_numpy(float)
        compositions.append({'grouping': group_col, 'group': group, 'n_households': len(frame),
                             'n_clusters': frame.grappe.nunique(), 'sample_share': len(frame)/len(hh),
                             'weighted_household_share': w.sum()/weights.sum()})
        for variable in ['pcexp', 'hhsize', 'hhweight', 's11q02']:
            summaries.append({'grouping': group_col, 'group': group, 'variable': variable,
                              **weighted_summary(frame[variable], w)})
        for variable in HOUSING:
            if variable=='s11q02':
                continue
            table = category_frequencies(frame[variable], w)
            table.insert(0, 'variable', variable);table.insert(0, 'group', group);table.insert(0, 'grouping', group_col)
            frequencies.append(table)
        for variable in asset_cols:
            x = binary.loc[index, variable].to_numpy(float); valid = np.isfinite(x)
            rate = np.average(x[valid], weights=w[valid]) if valid.any() else np.nan
            asset_rates.append({'grouping': group_col, 'group': group, 'variable': variable,
                                'n': len(x), 'n_valid': int(valid.sum()), 'n_code_1': int((x==1).sum()),
                                'rate_code_1': float(np.nanmean(x)) if valid.any() else np.nan,
                                'weighted_rate_code_1': rate,
                                'weighted_missing_or_invalid': w[~valid].sum()/w.sum()})
    summaries = pd.DataFrame(summaries);composition = pd.DataFrame(compositions)
    rates = pd.DataFrame(asset_rates); frequencies = pd.concat(frequencies, ignore_index=True)
    for name, table in [('statistiques_ponderees', summaries), ('composition_echantillon', composition),
                        ('frequences_logement', frequencies), ('frequences_actifs', rates)]:
        save_table(table, output, name)
    global_rates = rates.query('grouping=="ensemble"').copy()
    global_rates['rare_or_near_constant_1pct'] = (global_rates.weighted_rate_code_1 < .01) | (global_rates.weighted_rate_code_1 > .99)
    save_table(global_rates, output, 'diagnostic_actifs_rares')
    correlation = pd.DataFrame(np.nan, index=asset_cols, columns=asset_cols)
    for i, a in enumerate(asset_cols):
        for j in range(i, len(asset_cols)):
            b = asset_cols[j];value = weighted_correlation(binary[a], binary[b], weights)
            correlation.loc[a,b]=value;correlation.loc[b,a]=value
    correlation.rename_axis('variable').reset_index().to_csv(output/'tables/correlations_actifs_ponderees.csv', index=False)
    redundant = pd.DataFrame([{'variable_1': a,'variable_2': b,'weighted_phi': correlation.loc[a,b]}
                               for i,a in enumerate(asset_cols) for b in asset_cols[i+1:]
                               if np.isfinite(correlation.loc[a,b]) and abs(correlation.loc[a,b])>=.70],
                              columns=['variable_1','variable_2','weighted_phi'])
    save_table(redundant, output, 'redondances_actifs_seuil_070')
    outliers = []
    for variable in ['pcexp', 'hhsize', 'hhweight', 's11q02']:
        row = weighted_summary(hh[variable], weights);iqr = row['q75']-row['q25']
        x = pd.to_numeric(hh[variable],errors='coerce')
        flag = (x<row['q25']-1.5*iqr)|(x>row['q75']+1.5*iqr)
        outliers.append({'variable':variable,'lower_tukey':row['q25']-1.5*iqr,'upper_tukey':row['q75']+1.5*iqr,
                         'n_flagged':int(flag.sum()),'weighted_share_flagged':float(np.average(flag,weights=weights))})
    outliers=pd.DataFrame(outliers);save_table(outliers,output,'valeurs_extremes_a_examiner')
    # Graphiques : uniquement descriptifs, poids ménages et dénominateurs explicites.
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    valid = hh.pcexp.gt(0)&hh.pcexp.notna()
    ax[0].hist(np.log10(hh.loc[valid,'pcexp']),weights=weights[valid],bins=45,density=True,color='#287d8e')
    ax[0].set(xlabel='log10(pcexp), unité du fichier source',ylabel='Densité pondérée ménages',title='Distribution du bien-être')
    ax[1].hist(hh.hhweight,bins=45,color='#d19248');ax[1].set(xlabel='hhweight',ylabel='Nombre de ménages',title='Distribution des poids')
    save_figure(fig,output,'distributions')
    plot=global_rates.sort_values('weighted_rate_code_1');fig,ax=plt.subplots(figsize=(9,12))
    ax.barh(plot.variable,100*plot.weighted_rate_code_1,color='#287d8e')
    ax.set(xlabel='Part pondérée de réponses code 1 parmi codes 1/2 (%)',title='Actifs : 45 codes d’articles, libellés à valider',xlim=(0,100))
    save_figure(fig,output,'actifs')
    fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(correlation,vmin=-1,vmax=1,cmap='RdBu_r')
    ticks=np.arange(0,len(asset_cols),3);ax.set_xticks(ticks,[asset_cols[i].replace('article_','') for i in ticks],rotation=90);ax.set_yticks(ticks,[asset_cols[i].replace('article_','') for i in ticks]);ax.set(title='Corrélations pondérées des réponses binarisées',xlabel='Code article',ylabel='Code article');fig.colorbar(im,ax=ax,label='Corrélation phi pondérée')
    save_figure(fig,output,'correlations_actifs')
    milieu=rates.query('grouping=="milieu"').pivot(index='variable',columns='group',values='weighted_rate_code_1')
    if milieu.shape[1]>=2:
        ranked=(milieu.iloc[:,0]-milieu.iloc[:,1]).abs().nlargest(15).index
        fig,ax=plt.subplots(figsize=(10,6));(100*milieu.loc[ranked]).plot.barh(ax=ax)
        ax.set(xlabel='Part pondérée code 1 parmi codes valides (%)',title='Écarts entre milieux : 15 articles les plus contrastés')
        ax.legend(title='Code milieu (source)');save_figure(fig,output,'ecarts_milieu')
    region=summaries.query('grouping=="region" and variable=="pcexp"').sort_values('median')
    fig,ax=plt.subplots(figsize=(10,7));ax.barh(region.group,region['median'],color='#287d8e');ax.set(xlabel='Médiane pondérée de pcexp (unité source)',ylabel='Code région',title='Description par région — sans intervalle de sondage');save_figure(fig,output,'regions')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,var in zip(axes.flat,['s11q18','s11q20','s11q33','s11q54']):
        table=frequencies.query('grouping=="ensemble" and variable==@var')
        ax.bar(table.category,100*table.weighted_share,color='#287d8e');ax.set(title=HOUSING[var],xlabel='Code de modalité',ylabel='% pondéré ménages');ax.tick_params(axis='x',rotation=45)
    save_figure(fig,output,'logement')
    n_rare=int(global_rates.rare_or_near_constant_1pct.sum())
    electricity_codes=hh.s11q33.nunique(dropna=True)
    max_corr=float(redundant.weighted_phi.abs().max()) if len(redundant) else None
    sections=[
      ('Périmètre et limites',[
       f'Analyse descriptive initiale des {len(hh):,} ménages et {hh.grappe.nunique():,} grappes du socle welfare. Les fichiers bruts restent inchangés ; aucune ACP ni régression n’a été ajustée.',
       'Les poids utilisés sont hhweight. Les fréquences portent sur les ménages, pas sur les personnes. Les quantiles utilisent l’inverse de la distribution empirique pondérée. Les écarts-types sont descriptifs ; le Kish n résume la dispersion des poids et n’est pas une taille effective tenant compte des grappes et strates.',
       'Les codes des articles, régions, milieux et modalités sont conservés. Les intitulés de questions sont documentés, mais le dictionnaire complet des valeurs doit être validé avant recodage définitif. Le code 1 des actifs est analysé séparément du code 2 ; son interprétation comme possession est cohérente avec les quantités renseignées, sans remplacer la vérification des modalités.',
       'Cette EDA couvre l’échantillon disponible. Elle ne réserve pas un test jamais consulté et ne mesure aucune performance prédictive. Le protocole d’évaluation futur devra déclarer cette exploration préalable.'],None),
      ('Composition et pondération', ['Les parts pondérées ci-dessous estiment la composition des ménages sous les poids fournis. Aucun test de significativité ni intervalle de sondage n’est calculé.'], composition.query('grouping=="milieu"')),
      ('Qualité, non-réponse et sauts',[
       f'{int(missing.n_missing.sum())} cellules manquantes parmi les variables candidates examinées ; {invalid_asset_codes} codes d’actifs hors 1/2. Les fichiers CSV de diagnostic conservent les dénominateurs.',
       'Les quantités d’actifs manquantes après une non-possession sont des sauts attendus, pas des ménages à supprimer. Les lignes sans clés du fichier dérivé ménage restent hors du socle d’analyse ; les données originales ne sont pas corrigées automatiquement.'],inventory),
      ('Distributions et valeurs extrêmes',[
       'pcexp est décrit dans son unité source, sans nouvelle déflation. Les bornes de Tukey ci-dessous signalent des valeurs à examiner ; elles ne déclenchent ni suppression ni winsorisation.'],outliers),
      ('Variables patrimoniales et redondances',[
       f'{n_rare} des 45 articles ont une fréquence pondérée du code 1 inférieure à 1 % ou supérieure à 99 %. Ce seuil est un diagnostic, pas une règle automatique de sélection.',
       f'{len(redundant)} paires d’articles ont une corrélation pondérée absolue d’au moins 0,70. Elles doivent être examinées pour éviter de surpondérer un même domaine.',
       f's11q33 (électricité) présente {electricity_codes} modalités : un recodage binaire direct serait injustifié sans leur définition. Les catégories du logement ne seront pas traitées comme des nombres continus dans l’ACP.'],redundant.head(12)),
      ('Décisions avant préparation et ACP',[
       '1. Compléter les libellés et règles de saut à partir des questionnaires ; valider les recodages de chaque variable.',
       '2. Examiner la couverture des domaines logement/biens/services et la rareté par milieu ; éviter une sélection fondée uniquement sur la fréquence globale.',
       '3. Fixer le protocole des valeurs manquantes et extrêmes ; versionner une base de préparation distincte et un journal des exclusions.',
       '4. Définir les partitions spatiales et le statut de la cible (indice descriptif fixé ou construction entièrement dans l’apprentissage) avant sélection supervisée.',
       '5. Construire et diagnostiquer l’ACP seulement après ces décisions : contributions, variance expliquée, signe, robustesse, puis association patrimoine–consommation.'],None),
    ]
    figures=[('distributions','Bien-être et pondérations'),('actifs','Fréquences par code article'),('logement','Modalités du logement'),('correlations_actifs','Redondances des actifs'),('regions','Hétérogénéité régionale descriptive')]
    if milieu.shape[1]>=2:figures.append(('ecarts_milieu','Écarts descriptifs entre codes de milieu'))
    render_report(output,'Analyse exploratoire initiale — EHCVM Côte d’Ivoire 2021–2022',sections,figures,SOURCES)
    write_manifest(output,[archive_path],__file__,'ehcvm',{'n_households':len(hh),'n_clusters':int(hh.grappe.nunique()),'asset_rare_1pct':n_rare,'asset_pairs_abs_corr_070':len(redundant),'max_abs_redundant_corr':max_corr})
    return output/'rapport.html'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--output-dir',type=Path,default=EHCVM_OUTPUTS_DIR/'eda')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv);archive=(args.archive or default_archive()).expanduser().resolve()
    if args.dry_run:
        check_archive(archive);print('Entrées EDA EHCVM présentes ; aucune analyse exécutée.');return
    print(analysis(archive,args.output_dir.expanduser().resolve()))

if __name__=='__main__':
    main()
