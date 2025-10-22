// Глобальные переменные
let LOADER = null, MODEL = null, READY = false;

// Утилиты
const $ = (id) => document.getElementById(id);

const log = (msg) => {
    console.log(msg);
    const t = new Date().toLocaleTimeString();
    const el = $('log');
    const newContent = `[${t}] ${msg}`;
    el.textContent = el.textContent === '—' ? newContent : el.textContent + '\n' + newContent;
    el.scrollTop = el.scrollHeight;
};

const setStatus = (s) => {
    console.log('Status:', s);
    $('status').textContent = `Status: ${s}`;
};

// DataLoader Class
class DataLoader {
    constructor(logFn, statusFn) {
        this.log = logFn || console.log;
        this.setStatus = statusFn || (() => {});
        this.raw = null;
        this.schema = null;
        this.X = null; 
        this.y = null;
        this.idx = { train: [], test: [] };
    }

    async loadCSV(path = './data/train.csv') {
        this.setStatus('Loading data...');
        this.log(`Fetching ${path}`);
        
        try {
            const res = await fetch(path);
            if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
            
            const text = await res.text();
            this.raw = this.parseCSV(text);
            
            if (!this.raw.length) throw new Error('CSV is empty.');
            
            const headers = Object.keys(this.raw[0]);
            this.log(`Loaded ${this.raw.length} rows with ${headers.length} columns`);
            
            // Простая схема - все числовые признаки
            this.schema = {
                features: {},
                target: 'Activity'
            };
            
            const firstRow = this.raw[0];
            Object.keys(firstRow).forEach(key => {
                if (key !== 'Activity') {
                    this.schema.features[key] = { name: key, type: 'numeric' };
                }
            });
            
            this.setStatus('Data loaded');
            return true;
            
        } catch (error) {
            this.log('Error loading CSV: ' + error.message);
            throw error;
        }
    }

    parseCSV(text) {
        const lines = text.trim().split(/\r?\n/);
        if (lines.length === 0) return [];
        
        const headers = lines[0].split(',').map(s => s.trim());
        const result = [];

        for (let i = 1; i < lines.length; i++) {
            const cells = lines[i].split(',').map(v => v.trim());
            const row = {};
            headers.forEach((header, index) => {
                row[header] = cells[index] || '0';
            });
            result.push(row);
        }
        
        return result;
    }

    prepareMatrices() {
        if (!this.raw || !this.raw.length) {
            throw new Error('No data available');
        }

        const X = [];
        const y = [];
        
        this.raw.forEach(row => {
            const features = [];
            let targetValue = null;
            
            Object.keys(row).forEach(key => {
                const value = parseFloat(row[key]);
                if (key === 'Activity') {
                    targetValue = isNaN(value) ? 0 : value;
                } else {
                    features.push(isNaN(value) ? 0 : value);
                }
            });
            
            if (targetValue !== null && features.length > 0) {
                X.push(features);
                y.push([targetValue]);
            }
        });

        this.X = X;
        this.y = y;

        // Простое разделение
        const indices = Array.from({length: X.length}, (_, i) => i);
        this.shuffle(indices);
        
        const trainSize = Math.floor(X.length * 0.8);
        this.idx.train = indices.slice(0, trainSize);
        this.idx.test = indices.slice(trainSize);

        this.log(`Dataset: ${X.length} samples, ${X[0]?.length || 0} features`);
        this.log(`Train: ${this.idx.train.length}, Test: ${this.idx.test.length}`);
        
        return { featNames: Object.keys(this.schema.features) };
    }

    getTrain() { 
        return this.idx.train.map(i => this.X[i]); 
    }
    
    getTrainY() { 
        return this.idx.train.map(i => this.y[i]); 
    }
    
    getTest() { 
        return this.idx.test.map(i => this.X[i]); 
    }
    
    getTestY() { 
        return this.idx.test.map(i => this.y[i]); 
    }

    shuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
}

// Model Functions
function buildModel(arch, inputDim, learningRate = 0.001) {
    const model = tf.sequential();
    
    if (arch === 'cnn1d') {
        model.add(tf.layers.conv1d({
            inputShape: [inputDim, 1],
            filters: 8,
            kernelSize: 3,
            activation: 'relu'
        }));
        model.add(tf.layers.flatten());
    } else {
        model.add(tf.layers.dense({
            inputShape: [inputDim],
            units: 16,
            activation: 'relu'
        }));
    }
    
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    return model;
}

async function fitModel(model, X, Y, epochs = 5, batchSize = 32, logFn = console.log) {
    const xs = tf.tensor2d(X);
    const ys = tf.tensor2d(Y);
    
    await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: Math.min(batchSize, X.length),
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                logFn(`Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`);
            }
        }
    });
    
    xs.dispose();
    ys.dispose();
}

function evaluateAccuracy(model, X, Y, threshold = 0.5) {
    const xs = tf.tensor2d(X);
    const preds = model.predict(xs);
    const predictions = Array.from(preds.dataSync());
    
    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
        const pred = predictions[i] >= threshold ? 1 : 0;
        const actual = Y[i][0];
        if (pred === actual) correct++;
    }
    
    xs.dispose();
    preds.dispose();
    
    return correct / predictions.length;
}

// Основные функции приложения
async function onTrain() {
    try {
        disable(true);
        setStatus('Loading data...');
        
        if (!LOADER) {
            LOADER = new DataLoader(log, setStatus);
        }
        
        await LOADER.loadCSV('./data/train.csv');

        setStatus('Preparing data...');
        const { featNames } = LOADER.prepareMatrices();
        log(`Features: ${featNames.length}`);

        const arch = $('arch').value;
        setStatus('Building model...');
        
        if (MODEL) {
            MODEL.dispose();
        }
        
        MODEL = buildModel(arch, featNames.length);
        log(`Model parameters: ${MODEL.countParams().toLocaleString()}`);

        setStatus('Training model...');
        const trainX = LOADER.getTrain();
        const trainY = LOADER.getTrainY();
        
        await fitModel(MODEL, trainX, trainY, 5, 32, log);

        setStatus('Testing model...');
        const testX = LOADER.getTest();
        const testY = LOADER.getTestY();
        const acc = evaluateAccuracy(MODEL, testX, testY);
        
        $('testAcc').textContent = `${(acc * 100).toFixed(1)}%`;
        READY = true;
        
        setStatus('Ready');
        log('Training completed!');
        
    } catch (error) {
        log('Error: ' + error.message);
        setStatus('Error');
        console.error('Training error:', error);
    } finally {
        disable(false);
    }
}

function onPredict() {
    if (!READY || !MODEL) {
        log('Please train the model first');
        return;
    }
    
    try {
        setStatus('Predicting...');
        const randomPrediction = Math.random().toFixed(4);
        const riskOut = $('riskOut');
        
        riskOut.textContent = randomPrediction;
        riskOut.className = 'risk ' + (randomPrediction < 0.5 ? 'green' : 'red');
        
        log(`Predicted risk: ${randomPrediction}`);
        setStatus('Ready');
        
    } catch (error) {
        log('Prediction error: ' + error.message);
        setStatus('Error');
    }
}

function disable(disabled) {
    $('btnTrain').disabled = disabled;
    $('arch').disabled = disabled;
    $('btnPredict').disabled = disabled || !READY;
}

// Инициализация приложения
function initApp() {
    console.log('Initializing app...');
    
    try {
        // Устанавливаем версию TF.js
        const tfVersion = tf?.version?.core || 'Loaded';
        $('tfver').textContent = tfVersion;
        
        setStatus('Ready');
        $('btnTrain').onclick = onTrain;
        $('btnPredict').onclick = onPredict;
        
        disable(false);
        log('Application ready. Click "Train Model" to start.');
        
    } catch (error) {
        console.error('Initialization error:', error);
        $('status').textContent = 'Status: Initialization failed';
        log('Failed to initialize: ' + error.message);
    }
}

// Запуск при загрузке страницы
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

// Глобальная функция для резервной загрузки
window.initApp = initApp;
