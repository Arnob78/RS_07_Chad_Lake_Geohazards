/**
 * COMPLETE MULTI-SENSOR DATA EXPORT FOR LAKE CHAD BASIN (2000-2023)
 * NDVI, LST, Reflectance, Precipitation, Wind, Elevation, and Soil Data
 * 
 * @description: Exports multi-sensor annual composites for Lake Chad Basin (2000-2023)
 * @author: Arnob Bormudoi
 * @date: November 29, 2024
 * @repository: https://github.com/Arnob78/RS_07_Chad_Lake_Geohazards.git
 * 
 * 
 */

// Define study area - Lake Chad Basin
var studyArea = ee.FeatureCollection('projects/ee-arnob-bormudoi2024/assets/chad_lake_basin_whole');

print('=== COMPLETE DATA CHECK, DISPLAY & EXPORT FOR 2000-2023 ===');

var years = ee.List.sequence(2000, 2023);

/**
 * Creates annual composites for any Earth Engine ImageCollection
 * @param {string} collectionId - Earth Engine ImageCollection ID
 * @param {string} datasetName - Human-readable dataset name
 * @param {ee.Reducer} reducer - Reducer for temporal aggregation
 * @returns {ee.ImageCollection} Annual composites
 */
function createAnnualCollection(collectionId, datasetName, reducer) {
  print('Processing: ' + datasetName);
  
  var annualCollection = ee.ImageCollection.fromImages(
    years.map(function(year) {
      year = ee.Number(year);
      var start = ee.Date.fromYMD(year, 1, 1);
      var end = start.advance(1, 'year');
      
      var yearCollection = ee.ImageCollection(collectionId)
        .filterBounds(studyArea)
        .filterDate(start, end);
      
      var count = yearCollection.size();
      var composite = yearCollection.reduce(reducer).clip(studyArea);
      
      return composite
        .set('year', year)
        .set('images_count', count)
        .set('system:time_start', start.millis());
    })
  );
  
  // Check availability
  var availability = annualCollection.aggregate_array('images_count');
  print(datasetName + ' - Images per year:', availability);
  
  var yearsWithData = availability.map(function(count) {
    return ee.Number(count).gt(0);
  }).reduce(ee.Reducer.sum());
  
  print(datasetName + ' - Years with data: ' + yearsWithData.getInfo() + '/24');
  
  return annualCollection;
}

// ========== DATA COLLECTION ==========

// 1. MODIS NDVI - Vegetation Index
var annualNDVI = createAnnualCollection('MODIS/006/MOD13A1', 'MODIS NDVI', ee.Reducer.median());

// 2. MODIS LST - Land Surface Temperature
var annualLST = createAnnualCollection('MODIS/061/MOD11A2', 'MODIS LST', ee.Reducer.median());

// 3. MODIS Reflectance - Surface Reflectance Bands
var annualReflectance = createAnnualCollection('MODIS/061/MOD09A1', 'MODIS Reflectance', ee.Reducer.median());

// 4. CHIRPS Precipitation - Rainfall Data
var annualPrecip = createAnnualCollection('UCSB-CHG/CHIRPS/DAILY', 'CHIRPS Precipitation', ee.Reducer.sum());

// 5. ERA5 Wind - Wind Components
var annualWind = ee.ImageCollection.fromImages(
  years.map(function(year) {
    year = ee.Number(year);
    var start = ee.Date.fromYMD(year, 1, 1);
    var end = start.advance(1, 'year');
    
    var yearCollection = ee.ImageCollection('ECMWF/ERA5/DAILY')
      .filterBounds(studyArea)
      .filterDate(start, end)
      .select(['u_component_of_wind_10m', 'v_component_of_wind_10m']);
    
    var count = yearCollection.size();
    var composite = yearCollection.mean().clip(studyArea);
    
    return composite
      .set('year', year)
      .set('images_count', count)
      .set('system:time_start', start.millis());
  })
);
print('ERA5 Wind - Images per year:', annualWind.aggregate_array('images_count'));
print('ERA5 Wind - Years with data: ' + annualWind.aggregate_array('images_count')
  .map(function(count) { return ee.Number(count).gt(0); })
  .reduce(ee.Reducer.sum()).getInfo() + '/24');

// ========== DATA VISUALIZATION ==========

// DISPLAY BANDS FOR VISUAL INSPECTION
Map.centerObject(studyArea, 7);
Map.addLayer(studyArea, {color: 'FF0000', fillColor: '00000000', width: 3}, 'Study Area');

// Display sample years for each dataset
var ndvi2020 = annualNDVI.filter(ee.Filter.eq('year', 2020)).first();
Map.addLayer(ndvi2020.select('NDVI_median'), 
  {min: 0, max: 8000, palette: ['brown', 'yellow', 'green']}, 
  'NDVI 2020', true);

var lst2020 = annualLST.filter(ee.Filter.eq('year', 2020)).first();
Map.addLayer(lst2020.select('LST_Day_1km_median'), 
  {min: 12000, max: 16000, palette: ['blue', 'yellow', 'red']}, 
  'LST 2020', false);

var reflectance2020 = annualReflectance.filter(ee.Filter.eq('year', 2020)).first();
Map.addLayer(reflectance2020, 
  {bands: ['sur_refl_b02_median', 'sur_refl_b01_median', 'sur_refl_b04_median'], min: 0, max: 3000}, 
  'Reflectance 2020', false);

var precip2020 = annualPrecip.filter(ee.Filter.eq('year', 2020)).first();
Map.addLayer(precip2020, 
  {min: 0, max: 1000, palette: ['white', 'blue', 'darkblue']}, 
  'Precipitation 2020', false);

var wind2020 = annualWind.filter(ee.Filter.eq('year', 2020)).first();
Map.addLayer(wind2020.select('u_component_of_wind_10m'), 
  {min: -5, max: 5, palette: ['red', 'white', 'blue']}, 
  'U-Wind 2020', false);

// Show band names for reference
print('');
print('=== BAND INFORMATION ===');
print('NDVI Bands:', ndvi2020.bandNames().getInfo());
print('LST Bands:', lst2020.bandNames().getInfo());
print('Reflectance Bands:', reflectance2020.bandNames().getInfo());
print('Precipitation Bands:', precip2020.bandNames().getInfo());
print('Wind Bands:', wind2020.bandNames().getInfo());

// ========== DATA EXPORT ==========

print('');
print('=== SETTING UP EXPORTS ===');

// Export NDVI data (24 bands: NDVI_median_2000, NDVI_median_2001, ..., NDVI_median_2023)
Export.image.toDrive({
  image: annualNDVI.toBands(),
  description: 'LakeChad_NDVI_Annual_2000_2023',
  folder: 'GEE_Exports',
  region: studyArea.geometry(),
  scale: 500,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export LST data (24 bands: LST_Day_1km_median_2000, ..., LST_Day_1km_median_2023)
Export.image.toDrive({
  image: annualLST.toBands(),
  description: 'LakeChad_LST_Annual_2000_2023',
  folder: 'GEE_Exports',
  region: studyArea.geometry(),
  scale: 1000,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export Reflectance data (multiple bands per year × 24 years)
Export.image.toDrive({
  image: annualReflectance.toBands(),
  description: 'LakeChad_Reflectance_Annual_2000_2023',
  folder: 'GEE_Exports',
  region: studyArea.geometry(),
  scale: 500,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export Precipitation data (24 bands: precipitation_2000, ..., precipitation_2023)
Export.image.toDrive({
  image: annualPrecip.toBands(),
  description: 'LakeChad_Precipitation_Annual_2000_2023',
  folder: 'GEE_Exports',
  region: studyArea.geometry(),
  scale: 5000,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export Wind data (48 bands: u_wind_2000, v_wind_2000, ..., u_wind_2023, v_wind_2023)
Export.image.toDrive({
  image: annualWind.toBands(),
  description: 'LakeChad_Wind_Annual_2000_2023',
  folder: 'GEE_Exports',
  region: studyArea.geometry(),
  scale: 25000,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export static data
var srtm = ee.Image('USGS/SRTMGL1_003').clip(studyArea);
Export.image.toDrive({
  image: srtm,
  description: 'LakeChad_SRTM_Elevation',
  folder: 'GEE_Exports',
  region: studyArea.geometry(),
  scale: 30,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

var soilTexture = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02').clip(studyArea);
Export.image.toDrive({
  image: soilTexture,
  description: 'LakeChad_Soil_Texture',
  folder: 'GEE_Exports',
  region: studyArea.geometry(),
  scale: 250,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

print('');
print('=== EXPORT TASKS CREATED ===');
print('1. LakeChad_NDVI_Annual_2000_2023 - 24 bands (NDVI_median_2000 to NDVI_median_2023)');
print('2. LakeChad_LST_Annual_2000_2023 - 24 bands (LST_Day_1km_median_2000 to LST_Day_1km_median_2023)');
print('3. LakeChad_Reflectance_Annual_2000_2023 - Multiple bands × 24 years');
print('4. LakeChad_Precipitation_Annual_2000_2023 - 24 bands (precipitation_2000 to precipitation_2023)');
print('5. LakeChad_Wind_Annual_2000_2023 - 48 bands (U&V wind 2000-2023)');
print('6. LakeChad_SRTM_Elevation - Single band');
print('7. LakeChad_Soil_Texture - Single band');
print('');
print('Go to Tasks tab → Run all exports → Download GeoTIFF files');
print('Band naming: Year 2000 = band 1, Year 2023 = band 24');