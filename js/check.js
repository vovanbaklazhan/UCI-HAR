import * as tf from '@tensorflow/tfjs';

export function visualizeEDA(data) {
  const columns = [
    'tBodyAcc-mean-X', 'tBodyAcc-mean-Y', 'tBodyAcc-mean-Z', 
    'tBodyAcc-std-X', 'tBodyAcc-std-Y', 'tBodyAcc-std-Z', 
    'tBodyGyro-mean-X', 'tBodyGyro-mean-Y', 'tBodyGyro-mean-Z',
    'tBodyGyro-std-X', 'tBodyGyro-std-Y', 'tBodyGyro-std-Z'
  ];
  
  columns.forEach(col => {
    const values = data.map(d => d[col]).filter(v => !isNaN(v));
    const hist = getHistogram(values); // Создать гистограмму данных
    console.log(`Histogram of ${col}:`, hist);
  });
}

export function getHistogram(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const bins = 10;
  const binWidth = (max - min) / bins;
  const histogram = new Array(bins).fill(0);

  values.forEach(val => {
    const bin = Math.min(Math.floor((val - min) / binWidth), bins - 1);
    histogram[bin]++;
  });

  return histogram;
}

export function handleMissingData(data) {
  const columns = [
    'tBodyAcc-mean-X', 'tBodyAcc-mean-Y', 'tBodyAcc-mean-Z', 
    'tBodyAcc-std-X', 'tBodyAcc-std-Y', 'tBodyAcc-std-Z', 
    'tBodyGyro-mean-X', 'tBodyGyro-mean-Y', 'tBodyGyro-mean-Z',
    'tBodyGyro-std-X', 'tBodyGyro-std-Y', 'tBodyGyro-std-Z'
  ];
  
  columns.forEach(col => {
    const values = data.map(d => d[col]).filter(v => !isNaN(v));
    const median = getMedian(values);
    data.forEach(d => {
      if (isNaN(d[col])) {
        d[col] = median; 
      }
    });
  });
}

function getMedian(values) {
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 === 0 ? (values[mid - 1] + values[mid]) / 2 : values[mid];
}

export function checkCorrelations(features) {
  const correlationMatrix = tf.tensor(features).transpose().matMul(tf.tensor(features));
  console.log('Correlation Matrix:', correlationMatrix.arraySync());
}
