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
            this.log(`First few headers: ${headers.slice(0, 10).join(', ')}...`);
            
            // Найдем целевую колонку - попробуем разные варианты
            let targetColumn = this.findTargetColumn(headers);
            
            if (!targetColumn) {
                // Если не нашли Activity, используем последнюю колонку как целевую
                targetColumn = headers[headers.length - 1];
                this.log(`Using last column as target: "${targetColumn}"`);
            }
            
            this.log(`Using target column: "${targetColumn}"`);
            
            // Проверим значения в целевой колонке
            const targetValues = this.raw.map(row => row[targetColumn]).slice(0, 10);
            this.log(`First 10 target values: ${targetValues.join(', ')}`);
            
            // Создаем простую схему
            this.schema = {
                features: {},
                target: targetColumn
            };
            
            // Все колонки кроме target - features
            headers.forEach(key => {
                if (key !== targetColumn) {
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

    findTargetColumn(headers) {
        // Попробуем найти целевую колонку по разным именам
        const possibleTargets = ['Activity', 'activity', 'target', 'label', 'class'];
        
        for (const target of possibleTargets) {
            if (headers.includes(target)) {
                return target;
            }
        }
        
        // Попробуем найти по частичному совпадению
        for (const header of headers) {
            const headerLower = header.toLowerCase().replace(/"/g, '');
            if (headerLower.includes('activity') || 
                headerLower.includes('target') ||
                headerLower.includes('label') ||
                headerLower.includes('class')) {
                return header;
            }
        }
        
        return null;
    }

    parseCSV(text) {
        const lines = text.trim().split(/\r?\n/).filter(line => line.trim());
        if (lines.length === 0) return [];
        
        // Убираем кавычки из заголовков
        const headers = lines[0].split(',').map(s => s.trim().replace(/"/g, ''));
        const result = [];

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            
            // Убираем кавычки из данных
            const cells = line.split(',').map(v => v.trim().replace(/"/g, ''));
            const row = {};
            
            headers.forEach((header, index) => {
                // Если ячеек меньше чем заголовков, заполняем недостающие нулями
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
        
        this.raw.forEach((row, index) => {
            const features = [];
            let targetValue = null;
            let hasValidData = false;
            
            Object.keys(row).forEach(key => {
                const value = row[key];
                
                if (key === this.schema.target) {
                    // Target value - пробуем разные варианты парсинга
                    let num = parseFloat(value);
                    
                    // Если не число, пробуем преобразовать текстовые значения
                    if (isNaN(num)) {
                        const lowerValue = value.toLowerCase();
                        if (lowerValue === 'true' || lowerValue === 'yes' || lowerValue === '1') {
                            num = 1;
                        } else if (lowerValue === 'false' || lowerValue === 'no' || lowerValue === '0') {
                            num = 0;
                        } else {
                            // Пробуем извлечь число из строки
                            const match = value.match(/\d+/);
                            num = match ? parseInt(match[0]) : 0;
                        }
                    }
                    
                    if (!isNaN(num)) {
                        targetValue = num;
                        hasValidData = true;
                    }
                } else {
                    // Feature value
                    const num = parseFloat(value);
                    if (!isNaN(num)) {
                        features.push(num);
                        hasValidData = true;
                    } else {
                        features.push(0); // Заменяем NaN на 0
                    }
                }
            });
            
            if (targetValue !== null && features.length > 0 && hasValidData) {
                X.push(features);
                y.push(targetValue);
            } else {
                if (index < 5) { // Логируем только первые несколько ошибок
                    this.log(`Skipping row ${index} - invalid data. Target: ${targetValue}, Features: ${features.length}`);
                }
            }
        });

        this.log(`Processed ${X.length} valid samples out of ${this.raw.length} total`);
        
        if (X.length === 0) {
            // Покажем пример данных для отладки
            this.log(`First row sample: ${JSON.stringify(this.raw[0])}`);
            this.log(`Target column: ${this.schema.target}`);
            this.log(`Target value in first row: ${this.raw[0][this.schema.target]}`);
            throw new Error('No valid samples found after processing. Check data format.');
        }

        this.X = X;
        this.y = y;

        // Разделение на train/test
        const indices = Array.from({length: X.length}, (_, i) => i);
        this.shuffle(indices);
        
        const trainSize = Math.floor(X.length * 0.8);
        this.idx.train = indices.slice(0, trainSize);
        this.idx.test = indices.slice(trainSize);

        this.log(`Final dataset: ${X.length} samples, ${X[0].length} features`);
        this.log(`Train set: ${this.idx.train.length}, Test set: ${this.idx.test.length}`);
        
        // Логируем распределение целевых значений
        const targetCounts = y.reduce((acc, val) => {
            acc[val] = (acc[val] || 0) + 1;
            return acc;
        }, {});
        this.log(`Target distribution: ${JSON.stringify(targetCounts)}`);
        
        // Если все значения одинаковые, это проблема
        const uniqueValues = Object.keys(targetCounts);
        if (uniqueValues.length === 1) {
            this.log(`WARNING: All target values are ${uniqueValues[0]}. This may indicate a problem with data parsing.`);
        }
        
        return { 
            featNames: Object.keys(this.schema.features),
            dataInfo: {
                totalSamples: X.length,
                featuresCount: X[0].length,
                trainSamples: this.idx.train.length,
                testSamples: this.idx.test.length
            }
        };
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
    
    // Простая архитектура для избежания сложностей
    model.add(tf.layers.dense({
        inputShape: [inputDim],
        units: 32,
        activation: 'relu'
    }));
    
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    return model;
}

async function fitModel(model, X, Y, epochs = 5, batchSize = 32, logFn = console.log) {
    // Добавляем проверки данных
    if (!X || X.length === 0) {
        throw new Error('Training data X is empty');
    }
    if (!Y || Y.length === 0) {
        throw new Error('Training labels Y are empty');
    }
    
    logFn(`Training data shape: X=[${X.length}, ${X[0].length}], Y=[${Y.length}]`);
    
    try {
        // Преобразуем данные в правильный формат для TensorFlow
        const xArray = [];
        const yArray = [];
        
        for (let i = 0; i < X.length; i++) {
            xArray.push(X[i]);
            yArray.push([Y[i]]); // Преобразуем в 2D массив [[value], [value], ...]
        }
        
        // Создаем тензоры с явным указанием формы
        const xs = tf.tensor2d(xArray, [xArray.length, xArray[0].length]);
        const ys = tf.tensor2d(yArray, [yArray.length, 1]);
        
        logFn(`Tensor shapes - X: [${xs.shape}], Y: [${ys.shape}]`);
        
        await model.fit(xs, ys, {
            epochs: epochs,
            batchSize: Math.min(batchSize, X.length),
            validationSplit: 0.2,
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    logFn(`Epoch ${epoch + 1}/${epochs} - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}`);
                }
            }
        });
        
        xs.dispose();
        ys.dispose();
        
    } catch (error) {
        logFn(`Error during training: ${error.message}`);
        throw error;
    }
}

function predictOne(model, x) {
    try {
        const xs = tf.tensor2d([x], [1, x.length]);
        const pred = model.predict(xs);
        const result = pred.dataSync()[0];
        xs.dispose();
        pred.dispose();
        return result;
    } catch (error) {
        console.error('Prediction error:', error);
        return 0;
    }
}

function evaluateAccuracy(model, X, Y, threshold = 0.5) {
    if (!X || X.length === 0 || !Y || Y.length === 0) {
        console.error('Empty data in evaluateAccuracy');
        return 0;
    }
    
    try {
        // Преобразуем данные в правильный формат
        const xArray = [];
        const yArray = [];
        
        for (let i = 0; i < X.length; i++) {
            xArray.push(X[i]);
            yArray.push([Y[i]]);
        }
        
        const xs = tf.tensor2d(xArray, [xArray.length, xArray[0].length]);
        const preds = model.predict(xs);
        const predictions = Array.from(preds.dataSync());
        
        let correct = 0;
        for (let i = 0; i < predictions.length; i++) {
            const pred = predictions[i] >= threshold ? 1 : 0;
            const actual = Y[i];
            if (pred === actual) correct++;
        }
        
        const accuracy = correct / predictions.length;
        
        xs.dispose();
        preds.dispose();
        
        return accuracy;
    } catch (error) {
        console.error('Error in evaluateAccuracy:', error);
        return 0;
    }
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
        const { featNames, dataInfo } = LOADER.prepareMatrices();
        log(`Features count: ${featNames.length}`);
        log(`Data info: ${dataInfo.totalSamples} samples, ${dataInfo.featuresCount} features`);

        const arch = $('arch').value;
        setStatus('Building model...');
        
        if (MODEL) {
            MODEL.dispose();
        }
        
        MODEL = buildModel(arch, dataInfo.featuresCount);
        log(`Model parameters: ${MODEL.countParams().toLocaleString()}`);

        setStatus('Training model...');
        const trainX = LOADER.getTrain();
        const trainY = LOADER.getTrainY();
        
        log(`Starting training with ${trainX.length} samples`);
        await fitModel(MODEL, trainX, trainY, 5, 32, log);

        setStatus('Testing model...');
        const testX = LOADER.getTest();
        const testY = LOADER.getTestY();
        
        if (testX.length > 0) {
            const acc = evaluateAccuracy(MODEL, testX, testY);
            $('testAcc').textContent = `${(acc * 100).toFixed(1)}%`;
            log(`Test accuracy: ${(acc * 100).toFixed(1)}%`);
        } else {
            log('No test data available');
            $('testAcc').textContent = 'N/A';
        }
        
        READY = true;
        
        // Создаем простую форму для предсказания
        createSimplePredictionForm();
        
        // Исправляем ошибку - убираем обращение к несуществующему элементу
        if ($('simCard')) {
            $('simCard').style.opacity = '1';
        }
        
        setStatus('Ready');
        log('Training completed successfully!');
        
    } catch (error) {
        log('Error: ' + error.message);
        setStatus('Error');
        console.error('Training error:', error);
    } finally {
        disable(false);
    }
}

function createSimplePredictionForm() {
    const simGrid = $('simGrid');
    if (!simGrid) return;
    
    simGrid.innerHTML = '';
    
    if (!LOADER || !LOADER.schema) {
        simGrid.innerHTML = '<p>Model trained. Use predict button for random prediction.</p>';
        return;
    }
    
    const features = Object.keys(LOADER.schema.features);
    if (features.length > 0) {
        simGrid.innerHTML = `
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <p><strong>Model Info:</strong></p>
                <p>Features: ${features.length}</p>
                <p>Input dimension: ${LOADER.X[0].length}</p>
                <p>Click "Predict Risk" to test the model</p>
            </div>
        `;
    }
}

function onPredict() {
    if (!READY || !MODEL) {
        log('Please train the model first');
        return;
    }
    
    try {
        setStatus('Predicting...');
        
        // Создаем случайный вход на основе статистики тренировочных данных
        const testX = LOADER.getTest();
        if (testX.length > 0) {
            const randomIndex = Math.floor(Math.random() * testX.length);
            const randomSample = testX[randomIndex];
            const actualValue = LOADER.getTestY()[randomIndex];
            
            const prediction = predictOne(MODEL, randomSample);
            const riskOut = $('riskOut');
            
            if (riskOut) {
                riskOut.textContent = prediction.toFixed(4);
                
                // Определяем цвет риска
                let riskClass = 'green';
                if (prediction > 0.7) riskClass = 'red';
                else if (prediction > 0.3) riskClass = 'yellow';
                
                riskOut.className = 'risk ' + riskClass;
            }
            
            log(`Predicted risk: ${prediction.toFixed(4)} (actual: ${actualValue})`);
        } else {
            // Fallback - случайное предсказание
            const randomPrediction = Math.random().toFixed(4);
            const riskOut = $('riskOut');
            
            if (riskOut) {
                riskOut.textContent = randomPrediction;
                riskOut.className = 'risk ' + (randomPrediction < 0.5 ? 'green' : 'red');
            }
            
            log(`Random prediction: ${randomPrediction} (no test data available)`);
        }
        
        setStatus('Ready');
        
    } catch (error) {
        log('Prediction error: ' + error.message);
        setStatus('Error');
    }
}

function disable(disabled) {
    const btnTrain = $('btnTrain');
    const arch = $('arch');
    const btnPredict = $('btnPredict');
    
    if (btnTrain) btnTrain.disabled = disabled;
    if (arch) arch.disabled = disabled;
    if (btnPredict) btnPredict.disabled = disabled || !READY;
}

// Инициализация приложения
function initApp() {
    console.log('Initializing app...');
    
    try {
        // Устанавливаем версию TF.js
        const tfVersion = tf?.version?.core || 'Loaded';
        const tfverElement = $('tfver');
        if (tfverElement) {
            tfverElement.textContent = tfVersion;
        }
        
        setStatus('Ready');
        
        const btnTrain = $('btnTrain');
        const btnPredict = $('btnPredict');
        
        if (btnTrain) btnTrain.onclick = onTrain;
        if (btnPredict) btnPredict.onclick = onPredict;
        
        disable(false);
        log('Application ready. Click "Train Model" to start.');
        
    } catch (error) {
        console.error('Initialization error:', error);
        setStatus('Initialization failed');
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
