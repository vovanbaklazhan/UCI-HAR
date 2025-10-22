export class DataLoader {
  constructor(logFn, statusFn) {
    this.log = logFn || console.log;
    this.setStatus = statusFn || (() => {});
    this.raw = null;  // [{col:val,...}]
    this.schema = null;  // { features: {name,type,values?,stats?}, target:'Activity' }
    this.encoders = {};  // {catKey: [values...]}
    this.scaler = { type: 'minmax', stats: {} };  // by feature index
    this.X = null; this.y = null;
    this.idx = { train: [], test: [] };
    this.featNames = [];
  }

  // Загрузка CSV
  async loadCSV(path = './data/train.csv') {
    this.setStatus('loading data…');
    this.log(`Fetching ${path}`);
    
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
    
    const text = await res.text();
    this.raw = this.#parseCSV(text);
    if (!this.raw.length) throw new Error('CSV is empty.');

    // Логируем заголовки
    const headers = Object.keys(this.raw[0]);
    this.log('Raw Headers:', headers);

    // Очистим заголовки от пробелов и невидимых символов
    const cleanedHeaders = headers.map(header => header.trim().toLowerCase());
    this.log('Cleaned Headers:', cleanedHeaders);

    // Ищем целевой столбец (приводим к нижнему регистру для сравнения)
    const target = 'activity';
    const targetFound = cleanedHeaders.includes(target);
    
    if (!targetFound) {
      // Если не нашли, выводим доступные заголовки для отладки
      this.log('Available headers:', cleanedHeaders);
      throw new Error(`Target column "${target}" not found in the dataset. Available columns: ${cleanedHeaders.join(', ')}`);
    }

    this.#inferSchema(target);
    this.setStatus('data loaded');
    this.log(`Loaded rows=${this.raw.length}`);
  }

  // Парсинг CSV
  #parseCSV(text) {
    const [h, ...lines] = text.trim().split(/\r?\n/);
    const headers = h.split(',').map(s => s.trim());

    return lines.map(line => {
      const cells = line.split(',').map(v => v.trim());
      const o = {}; headers.forEach((k, i) => o[k] = cells[i] ?? '');
      return o;
    });
  }

  // Преобразование строки в число
  #num(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }

  // Определение схемы данных
  #inferSchema(target) {
    const features = {};
    
    // Используем оригинальные имена столбцов из raw данных
    const cols = Object.keys(this.raw[0]).filter(c => c.toLowerCase() !== target);
    
    for (const c of cols) {
      let type = 'numeric';
      // Приводим к нижнему регистру для сравнения
      const colLower = c.toLowerCase();
      if (['road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 'public_road', 'holiday', 'school_season']
          .map(s => s.toLowerCase()).includes(colLower)) {
        type = 'categorical';
      }
      if (['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
          .map(s => s.toLowerCase()).includes(colLower)) {
        type = 'numeric';
      }

      features[c] = { name: c, type };
    }
    
    // Статистика по числовым признакам и значениям категориальных
    for (const [k, f] of Object.entries(features)) {
      if (f.type === 'numeric') {
        const arr = this.raw.map(r => this.#num(r[k])).filter(Number.isFinite);
        if (arr.length === 0) {
          f.stats = { min: 0, max: 0, mean: 0, std: 0 };
        } else {
          const min = Math.min(...arr), max = Math.max(...arr);
          const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
          const std = Math.sqrt(arr.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(1, (arr.length - 1)));
          f.stats = { min, max, mean, std };
        }
      } else {
        let vals = [...new Set(this.raw.map(r => String(r[k])).filter(v => v !== '').map(String))];
        const lower = vals.map(v => v.toLowerCase());
        if (lower.every(v => v === 'true' || v === 'false')) { 
          f.type = 'boolean'; 
          vals = ['True', 'False']; 
        }
        f.values = vals.slice(0, 50);
      }
    }
    
    this.schema = { features, target };
  }

  // Подготовка матриц для обучения
  prepareMatrices() {
    this.encoders = {}; 
    this.featNames = [];
    
    // Строим энкодеры и featNames
    for (const [k, f] of Object.entries(this.schema.features)) {
      if (f.type === 'numeric' || f.type === 'boolean') { 
        this.featNames.push(k); 
      } else if (f.type === 'categorical') {
        this.encoders[k] = f.values || [];
        for (const v of this.encoders[k]) this.featNames.push(`${k}__${v}`);
      }
    }

    const X = [];
    const y = [];
    
    for (const r of this.raw) {
      const row = [];
      
      // Преобразуем данные в формате row
      for (const [k, f] of Object.entries(this.schema.features)) {
        if (f.type === 'numeric') {
          row.push(this.#num(r[k]));
        } else if (f.type === 'boolean') {
          row.push(String(r[k]).toLowerCase() === 'true' ? 1 : 0);
        } else {  // категориальные признаки — one-hot
          const cats = this.encoders[k] || [];
          for (const v of cats) row.push(String(r[k]) === v ? 1 : 0);
        }
      }
      X.push(row);
      
      // Получаем значение целевого столбца (используем оригинальное имя)
      const targetValue = r[Object.keys(r).find(key => key.toLowerCase() === this.schema.target)];
      y.push([Number(targetValue)]);
    }

    // Нормализация данных (MinMax)
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
      const colVals = X.map(r => r[c]).filter(Number.isFinite);
      if (colVals.length === 0) {
        this.scaler.stats[c] = { min: 0, max: 0, mean: 0, std: 0 };
      } else {
        const min = Math.min(...colVals), max = Math.max(...colVals);
        const mean = colVals.reduce((a, b) => a + b, 0) / Math.max(1, colVals.length);
        const std = Math.sqrt(colVals.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(1, (colVals.length - 1)));
        this.scaler.stats[c] = { min, max, mean, std };
      }
    }

    // Применяем MinMax scaling
    for (let i = 0; i < X.length; i++) {
      for (const c of numericIdx) {
        const st = this.scaler.stats[c];
        const v = X[i][c];
        if (!Number.isFinite(v)) {
          X[i][c] = 0;
          continue;
        }
        const d = (st.max - st.min) || 1;
        X[i][c] = (v - st.min) / d;
      }
    }

    this.X = X;
    this.y = y;

    // Разделение на train/test (80/20)
    const idx = [...Array(X.length).keys()];
    this.#shuffle(idx, 2025);
    const nTr = Math.floor(idx.length * 0.8);
    this.idx.train = idx.slice(0, nTr);
    this.idx.test = idx.slice(nTr);

    return { featNames: this.featNames };
  }

  // Функции для получения данных
  getTrain() { return this.idx.train.map(i => this.X[i]); }
  getTrainY() { return this.idx.train.map(i => this.y[i]); }
  getTest() { return this.idx.test.map(i => this.X[i]); }
  getTestY() { return this.idx.test.map(i => this.y[i]); }

  // Перемешивание индексов с использованием случайного сидирования
  #shuffle(a, seed = 123) {
    let s = seed;
    const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [a[i],a[j]]=[a[j],a[i]];
    }
  }
}
