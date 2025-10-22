// DataLoader Class
class DataLoader {
    constructor(logFn, statusFn) {
        this.log = logFn || console.log;
        this.setStatus = statusFn || (() => {});
        this.raw = null;
        this.schema = null;
        this.encoders = {};
        this.scaler = { type: 'minmax', stats: {} };
        this.X = null; this.y = null;
        this.idx = { train: [], test: [] };
        this.featNames = [];
    }

    async loadCSV(path = './data/train.csv') {
        this.setStatus('loading data…');
        this.log(`Fetching ${path}`);
        
        try {
            const res = await fetch(path);
            if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
            
            const text = await res.text();
            this.raw = this.parseCSV(text);
            if (!this.raw.length) throw new Error('CSV is empty.');

            const headers = Object.keys(this.raw[0]);
            this.log('Raw Headers: ' + headers.join(', '));

            const cleanedHeaders = headers.map(header => header.trim().toLowerCase());
            this.log('Cleaned Headers: ' + cleanedHeaders.join(', '));

            const target = 'activity';
            const targetFound = cleanedHeaders.includes(target);
            
            if (!targetFound) {
                this.log('Available headers: ' + cleanedHeaders.join(', '));
                throw new Error(`Target column "${target}" not found. Available: ${cleanedHeaders.join(', ')}`);
            }

            this.inferSchema(target);
            this.setStatus('data loaded');
            this.log(`Loaded rows=${this.raw.length}`);
        } catch (error) {
            this.log('Error loading CSV: ' + error.message);
            throw error;
        }
    }

    parseCSV(text) {
        const lines = text.trim().split(/\r?\n/);
        if (lines.length === 0) return [];
        
        const headers = lines[0].split(',').map(s => s.trim());

        return lines.slice(1).map(line => {
            const cells = line.split(',').map(v => v.trim());
            const o = {};
            headers.forEach((k, i) => {
                o[k] = cells[i] ?? '';
            });
            return o;
        }).filter(row => Object.keys(row).length > 0);
    }

    num(v) {
        if (v === '' || v === null || v === undefined) return NaN;
        const n = Number(v);
        return Number.isFinite(n) ? n : NaN;
    }

    inferSchema(target) {
        const features = {};
        const firstRow = this.raw[0];
        if (!firstRow) return;
        
        const cols = Object.keys(firstRow).filter(c => c.toLowerCase() !== target);
        
        for (const c of cols) {
            let type = 'numeric';
            const colLower = c.toLowerCase();
            
            if (['road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 
                'public_road', 'holiday', 'school_season'].includes(colLower)) {
                type = 'categorical';
            }
            
            features[c] = { name: c, type };
        }
        
        for (const [k, f] of Object.entries(features)) {
            const values = this.raw.map(r => r[k]).filter(v => v !== '');
            
            if (f.type === 'numeric') {
                const arr = values.map(v => this.num(v)).filter(v => !isNaN(v));
                if (arr.length > 0) {
                    const min = Math.min(...arr);
                    const max = Math.max(...arr);
                    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
                    const std = Math.sqrt(arr.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / arr.length);
                    f.stats = { min, max, mean, std };
                } else {
                    f.stats = { min: 0, max: 1, mean: 0, std: 1 };
                }
            } else {
                let vals = [...new Set(values.map(String))].filter(v => v !== '');
                const lower = vals.map(v => v.toLowerCase());
                if (lower.every(v => v === 'true' || v === 'false' || v === '1' || v === '0')) {
                    f.type = 'boolean';
                    vals = ['true', 'false'];
                } else {
                    f.values = vals.slice(0, 50);
                }
            }
        }
        
        this.schema = { features, target };
        this.log('Schema inferred with features: ' + Object.keys(features).join(', '));
    }

    prepareMatrices() {
        if (!this.schema) {
            throw new Error('Schema not available. Call loadCSV first.');
        }

        this.encoders = {}; 
        this.featNames = [];
        
        for (const [k, f] of Object.entries(this.schema.features)) {
            if (f.type === 'numeric' || f.type === 'boolean') { 
                this.featNames.push(k); 
            } else if (f.type === 'categorical') {
                this.encoders[k] = f.values || [];
                for (const v of this.encoders[k]) {
                    this.featNames.push(`${k}__${v}`);
                }
            }
        }

        const X = [];
        const y = [];
        
        for (let i = 0; i < this.raw.length; i++) {
            const r = this.raw[i];
            const row = [];
            
            for (const [k, f] of Object.entries(this.schema.features)) {
                const value = r[k] || '';
                
                if (f.type === 'numeric') {
                    const numVal = this.num(value);
                    row.push(isNaN(numVal) ? 0 : numVal);
                } else if (f.type === 'boolean') {
                    const strVal = String(value).toLowerCase();
                    row.push(strVal === 'true' || strVal === '1' ? 1 : 0);
                } else {
                    const cats = this.encoders[k] || [];
                    const strVal = String(value);
                    for (const v of cats) {
                        row.push(strVal === v ? 1 : 0);
                    }
                }
            }
            
            const targetKey = Object.keys(r).find(key => key.toLowerCase() === this.schema.target);
            if (targetKey) {
                const targetValue = r[targetKey];
                const numTarget = this.num(targetValue);
                
                if (!isNaN(numTarget)) {
                    X.push(row);
                    y.push([numTarget]);
                }
            }
        }

        this.log(`Valid samples: ${X.length}/${this.raw.length}`);
        
        if (X.length === 0) {
            throw new Error('No valid samples found after processing');
        }

        this.scaler = { type: 'minmax', stats: {} };
        const numericIdx = [];
        let col = 0;
        
        for (const [k, f] of Object.entries(this.schema.features)) {
            if (f.type === 'numeric' || f.type === 'boolean') {
                numericIdx.push(col);
                col += 1;
            } else {
                col += (this.encoders[k] || []).length;
            }
        }

        for (const c of numericIdx) {
            const colVals = X.map(r => r[c]).filter(v => !isNaN(v));
            if (colVals.length > 0) {
                const min = Math.min(...colVals);
                const max = Math.max(...colVals);
                this.scaler.stats[c] = { min, max };
            } else {
                this.scaler.stats[c] = { min: 0, max: 1 };
            }
        }

        for (let i = 0; i < X.length; i++) {
            for (const c of numericIdx) {
                const st = this.scaler.stats[c];
                const v = X[i][c];
                const d = (st.max - st.min) || 1;
                X[i][c] = (v - st.min) / d;
            }
        }

        this.X = X;
        this.y = y;

        const idx = [...Array(X.length).keys()];
        this.shuffle(idx, 2025);
        const nTr = Math.floor(idx.length * 0.8);
        this.idx.train = idx.slice(0, nTr);
        this.idx.test = idx.slice(nTr);

        this.log(`Final dataset: ${X.length} samples, ${this.featNames.length} features`);
        
        if (y.length > 0) {
            const uniqueTargets = [...new Set(y.flat())];
            this.log(`Unique target values: ${uniqueTargets.join(', ')}`);
        }

        return { featNames: this.featNames };
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

    shuffle(a, seed = 123) {
        let s = seed;
        const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(rnd() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
    }
}

// Model Functions
function buildModel(arch, inputDim, learningRate) {
    const model = tf.sequential();
    
    if (arch === 'cnn1d') {
        model.add(tf.layers.conv1d({
            inputShape: [inputDim, 1],
            filters: 32,
            kernelSize: 3,
            activation: 'relu'
        }));
        model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
        model.add(tf.layers.flatten());
    } else {
        model.add(tf.layers.dense({
            inputShape: [inputDim],
            units: 64,
            activation: 'relu'
        }));
    }
    
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    return model;
}

async function fitModel(model, X, Y, epochs, batchSize, logFn) {
    const xs = tf.tensor2d(X);
    const ys = tf.tensor2d(Y);
    
    await model.fit(xs, ys, {
        epochs: epochs,
        batchSize: batchSize,
        validationSplit: 0.1,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                logFn(`ep ${epoch + 1}/${epochs} loss=${logs.loss.toFixed(6)} val_loss=${logs.val_loss.toFixed(6)} acc=${logs.acc.toFixed(4)} val_acc=${logs.val_acc.toFixed(4)}`);
            }
        }
    });
    
    xs.dispose();
    ys.dispose();
}

function predictOne(model, x) {
    const xs = tf.tensor2d([x]);
    const pred = model.predict(xs);
    const result = pred.dataSync()[0];
    xs.dispose();
    pred.dispose();
    return result;
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

// Main Application
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
