"""EDA DHS des grappes disponibles ; les microdonnées ménages ne sont pas recréées."""
from pathlib import Path
import argparse
import sys

_ROOT=Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0,str(_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import DHS_PROCESSED_DIR, DHS_OUTPUTS_DIR
from src.utils.statistics import weighted_summary, category_frequencies
from src.utils.reporting import prepare_output, save_table, save_figure, render_report, write_manifest

INPUT=DHS_PROCESSED_DIR/'DHS_CI_2021_richesse_GPS_VIIRS_merged.csv'


def analysis(input_file,output):
    df=pd.read_csv(input_file)
    if df.cluster.isna().any() or df.cluster.duplicated().any():
        raise ValueError('Identifiants de grappes absents ou dupliqués')
    output=prepare_output(output)
    variables=['wealth_mean','wealth_mean_weighted','n_households','viirs_2u5r_mean','viirs_5all_mean','viirs_2u10r_mean']
    variables=[v for v in variables if v in df]
    summaries=[]
    for grouping in ['ensemble','urban_rural','adm1_name']:
        groups=[('tous',df)] if grouping=='ensemble' else list(df.groupby(grouping,dropna=False))
        for group,sub in groups:
            for variable in variables:
                summaries.append({'grouping':grouping,'group':group,'variable':variable,**weighted_summary(sub[variable],np.ones(len(sub)))})
    summaries=pd.DataFrame(summaries);save_table(summaries,output,'statistiques_grappes')
    missing=pd.DataFrame({'variable':df.columns,'n_missing':df.isna().sum().to_numpy(),'missing_rate':df.isna().mean().to_numpy()})
    save_table(missing,output,'manquants')
    composition=category_frequencies(df.urban_rural,np.ones(len(df))).rename(columns={'category':'milieu','share':'share_clusters','weighted_share':'share_equal_cluster_weights'})
    save_table(composition,output,'composition_grappes')
    correlations=df[variables].corr(method='spearman');correlations.rename_axis('variable').reset_index().to_csv(output/'tables/correlations_spearman.csv',index=False)
    viirs_diagnostic=[]
    for v in [c for c in df if c.startswith('viirs_') and c.endswith(('_mean','_coverage'))]:
        x=pd.to_numeric(df[v],errors='coerce')
        viirs_diagnostic.append({'variable':v,'n_missing':int(x.isna().sum()),'n_negative':int((x<0).sum()),'n_zero':int((x==0).sum()),'min':x.min(),'median':x.median(),'max':x.max()})
    save_table(pd.DataFrame(viirs_diagnostic),output,'qualite_viirs')
    diff=df.wealth_mean_weighted-df.wealth_mean
    diff_stats=pd.DataFrame([{'n':len(df),'n_abs_difference_gt_1e-8':int((diff.abs()>1e-8).sum()),'mean_absolute_difference':diff.abs().mean(),'max_absolute_difference':diff.abs().max(),'spearman':df[['wealth_mean','wealth_mean_weighted']].corr(method='spearman').iloc[0,1]}])
    save_table(diff_stats,output,'comparaison_cibles')
    outliers=[]
    for v in variables:
        x=df[v];q1=x.quantile(.25);q3=x.quantile(.75);iqr=q3-q1
        outliers.append({'variable':v,'lower_tukey':q1-1.5*iqr,'upper_tukey':q3+1.5*iqr,'n_flagged':int(((x<q1-1.5*iqr)|(x>q3+1.5*iqr)).sum())})
    save_table(pd.DataFrame(outliers),output,'valeurs_extremes')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(df.wealth_mean,bins=30,color='#287d8e');axes[0].set(xlabel='wealth_mean (échelle source)',ylabel='Nombre de grappes',title='Distribution de la cible utilisée')
    axes[1].hist(df.n_households,bins=25,color='#d19248');axes[1].set(xlabel='Ménages par grappe',ylabel='Nombre de grappes',title='Taille des grappes')
    save_figure(fig,output,'distributions')
    fig,ax=plt.subplots(figsize=(7,5))
    groups=list(df.groupby('urban_rural'))
    ax.boxplot([sub.wealth_mean.dropna() for _,sub in groups],tick_labels=[str(code) for code,_ in groups])
    ax.set(xlabel='Milieu (codes U/R du fichier)',ylabel='wealth_mean',title='Richesse agrégée par milieu — poids égaux par grappe')
    save_figure(fig,output,'milieux')
    fig,ax=plt.subplots(figsize=(8,5))
    for code,sub in groups:
        ax.scatter(np.log1p(sub.viirs_2u5r_mean.clip(lower=0)),sub.wealth_mean,s=15,alpha=.55,label=str(code))
    ax.set(xlabel='log(1 + max(VIIRS, 0)), affichage uniquement',ylabel='wealth_mean',title='Association descriptive richesse–luminosité');ax.legend(title='Milieu')
    save_figure(fig,output,'richesse_viirs')
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(df.wealth_mean,df.wealth_mean_weighted,s=12,alpha=.5)
    low=min(df.wealth_mean.min(),df.wealth_mean_weighted.min());high=max(df.wealth_mean.max(),df.wealth_mean_weighted.max())
    ax.plot([low,high],[low,high],linestyle='--',color='black');ax.set(xlabel='Moyenne simple',ylabel='Moyenne pondérée fournie',title='Deux cibles disponibles dans la base DHS')
    save_figure(fig,output,'cibles')
    region=summaries.query('grouping=="adm1_name" and variable=="wealth_mean"').sort_values('mean')
    fig,ax=plt.subplots(figsize=(10,6));ax.barh(region.group,region['mean'],color='#287d8e');ax.set(xlabel='Moyenne non pondérée entre grappes',title='Hétérogénéité descriptive par région');save_figure(fig,output,'regions')
    sections=[
      ('Périmètre',[
       f'EDA de {len(df)} grappes DHS 2021, à partir du fichier déjà agrégé. Elle ne remplace pas une EDA des microdonnées ménages ni la vérification de la construction du score original.',
       'Chaque grappe reçoit le même poids. Les effectifs ménages ne sont pas utilisés comme substituts aux poids de sondage. Les tableaux par région ne sont pas présentés comme des estimations officielles représentatives.',
       'Les résultats de modèles historiques ont déjà utilisé ces données. Aucun jeu de test intact n’est revendiqué après cette exploration. Les décisions issues de l’EDA devront être distinguées de l’évaluation confirmatoire future.'],None),
      ('Composition et qualité',[
       'Les variables absentes, extrêmes et radiances négatives sont décrites dans les tableaux exportés. Aucun enregistrement n’a été supprimé, aucune cible n’a été remplacée.'],composition),
      ('Cible de richesse : choix à documenter',[
       'Les scripts existants utilisent wealth_mean. Le fichier contient aussi wealth_mean_weighted. La présence des deux colonnes ne démontre pas à elle seule leur mode de construction ; il faut retracer les poids, l’échelle et l’agrégation à partir des données ménages sources.',
       'Le choix de la cible doit précéder la comparaison finale des modèles et être aligné avec l’estimand EHCVM. Les différences observées sont résumées ci-dessous.'],diff_stats),
      ('Descriptif de la richesse par milieu',[
       'Ces statistiques décrivent les grappes observées. Les écarts entre milieux sont associatifs ; ni leur causalité ni leur incertitude sous le plan de sondage ne sont estimées ici.'],summaries.query('grouping=="urban_rural" and variable=="wealth_mean"')[['group','n','mean','sd','median','min','max']]),
      ('Décisions à prendre avant nouvelle modélisation',[
       '1. Documenter l’origine du score et choisir entre moyenne simple et pondérée, avec analyse de sensibilité.',
       '2. Examiner la couverture VIIRS, les valeurs extrêmes et les voisinages ; ne pas supprimer automatiquement les zones riches ou sombres.',
       '3. Figer les partitions spatiales, les métriques et le modèle de référence avant de sélectionner les caractéristiques ou hyperparamètres.',
       '4. Ajouter une EDA des images et des caractéristiques CNN : qualité radiométrique, nuages, canaux, recadrage, distribution et redondance des caractéristiques. Les tests d’intégrité des images déjà existants ne remplacent pas cette analyse.',
       '5. Réexaminer les résultats historiques comme expériences préliminaires, puis procéder à une évaluation finale documentée.'],None),
    ]
    render_report(output,'Analyse exploratoire initiale — DHS Côte d’Ivoire 2021',sections,
                  [('distributions','Distributions des grappes'),('milieux','Hétérogénéité par milieu'),('richesse_viirs','Richesse et luminosité'),('cibles','Comparaison des cibles disponibles'),('regions','Moyennes des grappes par région')])
    write_manifest(output,[input_file],__file__,'dhs',{'n_clusters':len(df),'mean_absolute_target_difference':float(diff.abs().mean())})
    return output/'rapport.html'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DHS_OUTPUTS_DIR/'eda')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if not INPUT.is_file():raise FileNotFoundError(INPUT)
    if args.dry_run:print('Entrée EDA DHS présente ; aucune analyse exécutée.');return
    print(analysis(INPUT,args.output_dir.expanduser().resolve()))

if __name__=='__main__':main()
