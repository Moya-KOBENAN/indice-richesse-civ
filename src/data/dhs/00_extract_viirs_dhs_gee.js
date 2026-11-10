// ============================================================================
// Côte d'Ivoire DHS 2021 — Extraction des lumières nocturnes VIIRS
// Étape 1 de la phase géospatiale du projet
//
// Données d'entrée : 539 clusters DHS + score moyen de richesse.
// Produit VIIRS : NOAA/VIIRS/DNB/ANNUAL_V21, année 2021.
// Bande principale : average_masked (radiance DNB filtrée).
//
// Stratégies de buffer calculées en parallèle :
//   A) 2 km urbain / 5 km rural  -> buffer principal
//   B) 5 km pour tous           -> analyse de sensibilité (DHS, raster continu)
//   C) 2 km urbain / 10 km rural -> sensibilité conservatrice côté rural
// ============================================================================

// 1) Clusters DHS
// Importer les grappes autorisées dans un asset privé ; ne pas publier cet asset.
// Champs attendus : cluster, wealth_mean, wealth_mean_weighted,
// n_households, urban_rural, adm1_name, dhs_region, et géométrie.
var clusters = ee.FeatureCollection('projects/YOUR_PROJECT/assets/PRIVATE_CLUSTERS');

print('Nombre de clusters DHS', clusters.size());

// 2) VIIRS annuel 2021
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/ANNUAL_V21')
  .filterDate('2021-01-01', '2022-01-01')
  .first();

print('VIIRS 2021', viirs);

var rad = viirs.select('average_masked').rename('viirs_rad');
var coverage = viirs.select('cf_cvg').rename('viirs_cf_cvg');

// On garde la radiance >= 0 pour le benchmark principal.
var radNonNeg = rad.updateMask(rad.gte(0));

// 3) Fonction d'extraction des statistiques dans un buffer
function extractStats(feature, radiusMeters, prefix) {
  var zone = feature.geometry().buffer(radiusMeters);

  var radStats = radNonNeg.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.median(), '', true)
      .combine(ee.Reducer.max(), '', true)
      .combine(ee.Reducer.stdDev(), '', true),
    geometry: zone,
    scale: 500,
    bestEffort: true,
    maxPixels: 1e8
  });

  var covStats = coverage.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: zone,
    scale: 500,
    bestEffort: true,
    maxPixels: 1e8
  });

  return feature
    .set(prefix + '_radius_m', radiusMeters)
    .set(prefix + '_mean', radStats.get('viirs_rad_mean'))
    .set(prefix + '_median', radStats.get('viirs_rad_median'))
    .set(prefix + '_max', radStats.get('viirs_rad_max'))
    .set(prefix + '_stddev', radStats.get('viirs_rad_stdDev'))
    .set(prefix + '_cf_cvg_mean', covStats.get('viirs_cf_cvg'));
}

// 4) Extraction pour les 3 stratégies
var results = clusters.map(function(f) {
  var ur = ee.String(f.get('urban_rural'));

  // A. Principal : 2 km urbain / 5 km rural
  var rA = ee.Number(ee.Algorithms.If(ur.equals('U'), 2000, 5000));
  var out = extractStats(f, rA, 'viirs_2u5r');

  // B. Sensibilité : 5 km pour tout le monde
  out = extractStats(out, 5000, 'viirs_5all');

  // C. Sensibilité : 2 km urbain / 10 km rural
  var rC = ee.Number(ee.Algorithms.If(ur.equals('U'), 2000, 10000));
  out = extractStats(out, rC, 'viirs_2u10r');

  return out;
});

print('Aperçu des résultats', results.limit(5));

// 5) Carte de contrôle
Map.centerObject(clusters, 7);
Map.addLayer(
  radNonNeg,
  {min: 0, max: 25},
  'VIIRS 2021 average_masked'
);
Map.addLayer(
  clusters,
  {color: 'red'},
  'Clusters DHS'
);

// 6) Export CSV vers Google Drive
Export.table.toDrive({
  collection: results,
  description: 'DHS_CI_2021_VIIRS_clusters',
  fileNamePrefix: 'DHS_CI_2021_clusters_richesse_GPS_VIIRS',
  fileFormat: 'CSV',
  selectors: [
    'cluster',
    'wealth_mean',
    'wealth_mean_weighted',
    'n_households',
    'urban_rural',
    'adm1_name',
    'dhs_region',

    'viirs_2u5r_radius_m',
    'viirs_2u5r_mean',
    'viirs_2u5r_median',
    'viirs_2u5r_max',
    'viirs_2u5r_stddev',
    'viirs_2u5r_cf_cvg_mean',

    'viirs_5all_radius_m',
    'viirs_5all_mean',
    'viirs_5all_median',
    'viirs_5all_max',
    'viirs_5all_stddev',
    'viirs_5all_cf_cvg_mean',

    'viirs_2u10r_radius_m',
    'viirs_2u10r_mean',
    'viirs_2u10r_median',
    'viirs_2u10r_max',
    'viirs_2u10r_stddev',
    'viirs_2u10r_cf_cvg_mean',

    '.geo'
  ]
});
