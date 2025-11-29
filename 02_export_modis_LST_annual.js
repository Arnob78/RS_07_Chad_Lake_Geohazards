/**
 * LAKE CHAD BASIN - MODIS LST DATA DOWNLOAD PER YEAR (2000-2023)
 * Annual Land Surface Temperature data extraction from MODIS
 * 
 * @description: Extracts annual MODIS LST composites for Lake Chad Basin (2000-2023)
 * @author: Arnob Bormudoi
 * @date: November 29, 2024
 * @repository: https://github.com/Arnob78/RS_07_Chad_Lake_Geohazards.git
 * 
 * 
 */

// Define study area - Lake Chad Basin
var studyArea = ee.FeatureCollection('projects/ee-arnob-bormudoi2024/assets/chad_lake_basin_whole');

// Center map on study area
Map.centerObject(studyArea, 7);

// Display study area
Map.addLayer(studyArea, {color: 'FF0000', fillColor: '00000000', width: 3}, 'Lake Chad Basin');

// Years to process
var years = ee.List.sequence(2000, 2023);

print('=== PROCESSING MODIS LST DATA FOR 2000-2023 ===');

/**
 * Creates annual LST composites from MODIS data
 * @returns {ee.ImageCollection} Annual LST composites
 */
function createAnnualLSTComposites() {
  var annualLSTCollection = ee.ImageCollection.fromImages(
    years.map(function(year) {
      year = ee.Number(year);
      var start = ee.Date.fromYMD(year, 1, 1);
      var end = start.advance(1, 'year');
      
      // Load MODIS LST data for the year
      var yearCollection = ee.ImageCollection('MODIS/061/MOD11A2')
        .filterBounds(studyArea)
        .filterDate(start, end);
      
      var count = yearCollection.size();
      var composite = yearCollection.median().clip(studyArea);
      
      return composite
        .set('year', year)
        .set('images_count', count)
        .set('system:time_start', start.millis());
    })
  );
  
  return annualLSTCollection;
}

// Create annual LST collection
var annualLSTCollection = createAnnualLSTComposites();

// Convert to list for individual processing
var lstList = annualLSTCollection.toList(annualLSTCollection.size());

// Check data availability
var availability = annualLSTCollection.aggregate_array('images_count');
print('MODIS LST - Images per year:', availability);

var yearsWithData = availability.map(function(count) {
  return ee.Number(count).gt(0);
}).reduce(ee.Reducer.sum());

print('MODIS LST - Years with data: ' + yearsWithData.getInfo() + '/24');

// Display sample year for visual inspection
var lst2020 = annualLSTCollection.filter(ee.Filter.eq('year', 2020)).first();
Map.addLayer(lst2020.select('LST_Day_1km'), {
  min: 13000,
  max: 16000,
  palette: ['blue', 'yellow', 'red']
}, 'LST Day 2020');

// Show band names for reference
print('');
print('=== LST BAND INFORMATION ===');
print('LST Bands:', lst2020.bandNames().getInfo());

// Calculate and display LST statistics
print('');
print('=== LST STATISTICS ===');
var lstStats = annualLSTCollection.map(function(image) {
  var meanDay = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea.geometry(),
    scale: 1000,
    maxPixels: 1e9
  }).get('LST_Day_1km');
  
  var meanNight = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea.geometry(),
    scale: 1000,
    maxPixels: 1e9
  }).get('LST_Night_1km');
  
  return ee.Feature(null, {
    'year': image.get('year'),
    'mean_lst_day': meanDay,
    'mean_lst_night': meanNight,
    'images_count': image.get('images_count')
  });
});

var statsList = lstStats.reduceColumns({
  reducer: ee.Reducer.toList(4),
  selectors: ['year', 'mean_lst_day', 'mean_lst_night', 'images_count']
}).get('list');

print('Yearly LST Statistics:', statsList.getInfo());

// SET UP INDIVIDUAL YEAR EXPORTS
print('');
print('=== SETTING UP INDIVIDUAL YEAR LST EXPORTS ===');

// Export each year as a separate complete GeoTIFF file
for (var i = 0; i < 24; i++) {
  var year = 2000 + i;
  var lstImage = ee.Image(lstList.get(i));
  
  // Export Daytime and Nighttime LST
  Export.image.toDrive({
    image: lstImage.select(['LST_Day_1km', 'LST_Night_1km']),
    description: 'LakeChad_LST_' + year + '_SINGLE_FILE',
    folder: 'GEE_Exports',
    region: studyArea.geometry(),
    scale: 1000,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    crs: 'EPSG:4326',
    formatOptions: {
      cloudOptimized: false // Disable internal tiling
    }
  });
  
  print('Export task created for year: ' + year + ' - SINGLE FILE');
}

// Export LST statistics as CSV
Export.table.toDrive({
  collection: lstStats,
  description: 'LakeChad_LST_Statistics_2000_2023',
  folder: 'GEE_Exports',
  fileFormat: 'CSV'
});

print('');
print('=== EXPORT TASKS CREATED ===');
print('MAIN EXPORTS (24 files):');
print('- LakeChad_LST_2000_SINGLE_FILE to LakeChad_LST_2023_SINGLE_FILE');
print('- 1000m resolution, single GeoTIFF files');
print('- Each file contains: LST_Day_1km and LST_Night_1km bands');
print('- No tiling, no mosaicking required');
print('');
print('STATISTICS (1 file):');
print('- LakeChad_LST_Statistics_2000_2023.csv');
print('');
print('DOWNLOAD INSTRUCTIONS:');
print('1. Go to Tasks tab in GEE');
print('2. Run all "LakeChad_LST_XXXX_SINGLE_FILE" tasks');
print('3. Each year will download as one complete GeoTIFF file');
print('4. Files contain both daytime and nighttime LST data');