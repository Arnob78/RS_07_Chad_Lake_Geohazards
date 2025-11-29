/**
 * COMPLETE DATA CHECK, DISPLAY & EXPORT FOR 2000-2023 - NDVI ONLY
 * Single files per year - No tiling required
 * 
 * @description: Exports annual MODIS NDVI composites for Lake Chad Basin (2000-2023)
 * @author: Arnob Bormudoi
 * @date: November 29, 2024
 * @repository: https://github.com/Arnob78/RS_07_Chad_Lake_Geohazards.git
 * 
 * 
 */

// Define study area - Lake Chad Basin
var studyArea = ee.FeatureCollection('projects/ee-arnob-bormudoi2024/assets/chad_lake_basin_whole');

print('=== NDVI DATA CHECK, DISPLAY & EXPORT FOR 2000-2023 ===');

// Define analysis period
var years = ee.List.sequence(2000, 2023);

/**
 * Creates annual NDVI composites from MODIS data
 * @returns {ee.ImageCollection} Annual NDVI composites
 */
function createAnnualNDVI() {
  print('Processing: MODIS NDVI');
  
  var annualNDVI = ee.ImageCollection.fromImages(
    years.map(function(year) {
      year = ee.Number(year);
      var start = ee.Date.fromYMD(year, 1, 1);
      var end = start.advance(1, 'year');
      
      // Load MODIS NDVI data for the year
      var yearCollection = ee.ImageCollection('MODIS/006/MOD13A1')
        .filterBounds(studyArea)
        .filterDate(start, end);
      
      var count = yearCollection.size();
      var composite = yearCollection.reduce(ee.Reducer.median()).clip(studyArea);
      
      return composite
        .set('year', year)
        .set('images_count', count)
        .set('system:time_start', start.millis());
    })
  );
  
  // Check data availability
  var availability = annualNDVI.aggregate_array('images_count');
  print('MODIS NDVI - Images per year:', availability);
  
  var yearsWithData = availability.map(function(count) {
    return ee.Number(count).gt(0);
  }).reduce(ee.Reducer.sum());
  
  print('MODIS NDVI - Years with data: ' + yearsWithData.getInfo() + '/24');
  
  return annualNDVI;
}

// Create NDVI collection
var annualNDVI = createAnnualNDVI();

// DISPLAY NDVI FOR VISUAL INSPECTION
Map.centerObject(studyArea, 7);
Map.addLayer(studyArea, {color: 'FF0000', fillColor: '00000000', width: 3}, 'Study Area');

// Display sample years for NDVI
var ndvi2020 = annualNDVI.filter(ee.Filter.eq('year', 2020)).first();
Map.addLayer(ndvi2020.select('NDVI_median'), 
  {min: 0, max: 8000, palette: ['brown', 'yellow', 'green']}, 
  'NDVI 2020');

// Show band names for reference
print('');
print('=== NDVI BAND INFORMATION ===');
print('NDVI Bands:', ndvi2020.bandNames().getInfo());

// Calculate and display NDVI statistics
print('');
print('=== NDVI STATISTICS ===');
var ndviStats = annualNDVI.map(function(image) {
  var mean = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea.geometry(),
    scale: 500,
    maxPixels: 1e9
  });
  return ee.Feature(null, {
    'year': image.get('year'),
    'mean_ndvi': mean.get('NDVI_median'),
    'images_count': image.get('images_count')
  });
});

var statsList = ndviStats.reduceColumns({
  reducer: ee.Reducer.toList(3),
  selectors: ['year', 'mean_ndvi', 'images_count']
}).get('list');

print('Yearly NDVI Statistics:', statsList.getInfo());

// SET UP INDIVIDUAL YEAR EXPORTS - EACH AS SINGLE COMPLETE FILE
print('');
print('=== SETTING UP INDIVIDUAL YEAR EXPORTS (NO TILING) ===');

// Export each year as a separate complete GeoTIFF file
for (var i = 0; i < 24; i++) {
  var year = 2000 + i;
  var ndviImage = annualNDVI.filter(ee.Filter.eq('year', year)).first();
  
  // Force single file export for each year
  Export.image.toDrive({
    image: ndviImage.select('NDVI_median'),
    description: 'LakeChad_NDVI_' + year + '_SINGLE_FILE',
    folder: 'GEE_Exports',
    region: studyArea.geometry(),
    scale: 500,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    crs: 'EPSG:4326',
    formatOptions: {
      cloudOptimized: false // Disable internal tiling
    }
  });
  
  print('Export task created for year: ' + year + ' - SINGLE FILE');
}

// Also create a backup option with lower resolution in case any year still tiles
print('');
print('=== CREATING BACKUP EXPORTS (LOWER RESOLUTION - GUARANTEED SINGLE FILES) ===');

for (var i = 0; i < 24; i++) {
  var year = 2000 + i;
  var ndviImage = annualNDVI.filter(ee.Filter.eq('year', year)).first();
  
  // Backup export with lower resolution to absolutely guarantee single file
  Export.image.toDrive({
    image: ndviImage.select('NDVI_median'),
    description: 'LakeChad_NDVI_' + year + '_1000m_SINGLE',
    folder: 'GEE_Exports',
    region: studyArea.geometry(),
    scale: 1000, // Lower resolution to ensure single file
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    crs: 'EPSG:4326'
  });
  
  print('Backup export created for year: ' + year + ' - 1000m resolution');
}

// Export NDVI statistics as CSV
Export.table.toDrive({
  collection: ndviStats,
  description: 'LakeChad_NDVI_Statistics_2000_2023',
  folder: 'GEE_Exports',
  fileFormat: 'CSV'
});

print('');
print('=== EXPORT TASKS CREATED ===');
print('MAIN EXPORTS (24 files):');
print('- LakeChad_NDVI_2000_SINGLE_FILE to LakeChad_NDVI_2023_SINGLE_FILE');
print('- 500m resolution, optimized for single files');
print('- Each year = one complete GeoTIFF, no parts, no mosaicking');
print('');
print('BACKUP EXPORTS (24 files):');
print('- LakeChad_NDVI_2000_1000m_SINGLE to LakeChad_NDVI_2023_1000m_SINGLE');
print('- 1000m resolution, guaranteed single files');
print('- Use if main exports still create multiple tiles');
print('');
print('STATISTICS (1 file):');
print('- LakeChad_NDVI_Statistics_2000_2023.csv');
print('');
print('DOWNLOAD INSTRUCTIONS:');
print('1. Go to Tasks tab');
print('2. Run all "LakeChad_NDVI_XXXX_SINGLE_FILE" tasks first (500m resolution)');
print('3. Check if each downloads as ONE file (not multiple parts)');
print('4. If any year creates multiple tiles, run its backup version instead');
print('5. You will get 24 individual GeoTIFF files - one for each year');
print('');
print('Each file contains: Single band "NDVI_median" for that specific year');
print('No stripes, no oblique boundaries, no mosaicking needed!');