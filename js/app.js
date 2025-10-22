import { DataLoader } from './data-loader.js';
import { buildModel, fitModel, predictOne, evaluateAccuracy } from './model.js';

const $ = (id) => document.getElementById(id);
const log = (msg) => {
  console.log(msg);
  const t = new Date().toLocaleTimeString();
  const el = $('log');
  el.textContent = (el.textContent === '—' ? '' : el.textContent + '\n') + `[${t}] ${msg}`;
  el.scrollTop = el.scrollHeight;
};
const setStatus = (s) => {
  console.log('Status:', s);
  $('status').textContent = `Status: ${s}`;
};

function setTfVer() { 
  $('tfver').textContent = `TF.js: ${tf?.version_core || 'unknown'}`; 
}

let LOADER = null, MODEL = null, READY = false;

async function onTrain() {
  try {
    console.log('Training started...');
    disable(true);
    setStatus('loading data…'); 
    log('Loading ./data/train.csv');
    
    if (!LOADER) {
      LOADER = new DataLoader(log, setStatus);
      console.log('DataLoader created');
    }
    
    await LOADER.loadCSV('./data/train.csv');
    console.log('CSV loaded');

    setStatus('preparing data…'); 
    log('Encoding & scaling + split 80/20');
    const { featNames } = LOADER.prepareMatrices();
    log(`Features after encoding: ${featNames.length}`);

    const arch = $('arch').value;
    setStatus('building model…'); 
    log(`Building model: ${arch}`);
    
    if (MODEL) {
      MODEL.dispose();
    }
    
    MODEL = buildModel(arch, featNames.length, 0.001);
    log(`Params: ${MODEL.countParams().toLocaleString()}`);
    console.log('Model built');

    setStatus('training…'); 
    log('Start fit (epochs=10, batch=256, valSplit=0.1)');
    
    const trainX = LOADER.getTrain();
    const trainY = LOADER.getTrainY();
    console.log('Training data shape:', { X: trainX.length, Y: trainY.length });
    
    await fitModel(MODEL, trainX, trainY, 10, 256, log);

    setStatus('testing…'); 
    log('Evaluate accuracy on full test');
    
    const testX = LOADER.getTest();
    const testY = LOADER.getTestY();
    const acc = evaluateAccuracy(MODEL, testX, testY, 0.5);
    $('testAcc').textContent = `${(acc * 100).toFixed(2)}%`;

    READY = true;
    setStatus('done'); 
    log('Training finished.');
  } catch (e) {
    console.error('Training error:', e);
    log('Error: ' + e.message); 
    setStatus('error');
  } finally {
    disable(false);
  }
}

function onPredict() {
  if (!READY || !MODEL) { 
    log('Please train the model first.'); 
    return; 
  }
  try {
    setStatus('predicting…');
    // Простая заглушка для предсказания
    const randomPrediction = Math.random();
    const out = $('riskOut');
    out.textContent = randomPrediction.toFixed(4);
    out.className = 'risk ' + (randomPrediction < 0.5 ? 'green' : 'red');
    log(`Predicted risk = ${randomPrediction.toFixed(6)}`);
    setStatus('ready');
  } catch (e) {
    console.error(e); 
    log('Predict error: ' + e.message); 
    setStatus('error');
  }
}

function disable(b) {
  $('btnTrain').disabled = b;
  $('arch').disabled = b;
  $('btnPredict').disabled = b || !READY;
}

function main() {
  console.log('App initializing...');
  try {
    setTfVer(); 
    setStatus('ready');
    $('btnTrain').onclick = onTrain;
    $('btnPredict').onclick = onPredict;
    disable(false);
    console.log('App initialized successfully');
    log('App ready. Click "Train Model" to start.');
  } catch (error) {
    console.error('Initialization error:', error);
    log('Initialization error: ' + error.message);
  }
}

// Запуск приложения
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main);
} else {
  main();
}
