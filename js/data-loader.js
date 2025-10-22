export class DataLoader {
  constructor(logFn, statusFn) {
    this.log = logFn || console.log;
    this.setStatus = statusFn || (() => {});
    this.raw = null;
    this.schema = null;
    this.encoders = {};
    this.scaler = { type: 'minmax', stats: {} };
    this.X = null;
    this.y = null;
    this.idx = { train: [], test: [] };
    this.featNames = [];
  }

  async loadCSV(paths = [
    'https://vovanbaklazhan.github.io/UCI-HAR/data/train_1.csv',
    'https://vovanbaklazhan.github.io/UCI-HAR/data/train_2.csv',
    'https://vovanbaklazhan.github.io/UCI-HAR/data/train_3.csv'
  ]) {
    this.setStatus('loading data…');
    this.log(`Fetching ${paths.join(', ')}`);
    this.raw = [];

    // Загружаем данные из всех файлов
    for (let path of paths) {
      try {
        const res = await fetch(path);
        if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
        const text = await res.text();
        const data = this.#parseCSV(text);
        if (!data.length) throw new Error(`CSV ${path} is empty.`);
        this.raw = [...this.raw, ...data]; // Объединяем данные из всех файлов
        this.log(`Loaded ${data.length} rows from ${path}`);
      } catch (e) {
        this.log(`Error loading ${path}: ${e.message}`);
        throw e;
      }
    }

    this.#inferSchema();
    this.setStatus('data loaded');
    this.log(`Total rows loaded: ${this.raw.length}`);
  }

  #parseCSV(text) {
    const [h, ...lines] = text.trim().split(/\r?\n/);
    const headers = h.split(',').map(s => s.trim());
    return lines.map(line => {
      const cells = line.split(',').map(v => v.trim());
      const o = {};
      headers.forEach((k, i) => o[k] = cells[i] ?? '');
      return o;
    });
  }

  #num(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }

  #inferSchema() {
    const target = 'Activity'; // Целевая переменная, которая соответствует активности
    if (!(target in this.raw[0])) throw new Error(`Target "${target}" not found`);
    const features = {};
    const cols = Object.keys(this.raw[0]).filter(c => c !== target); // Исключаем целевой признак

    for (const c of cols) {
      let type = 'numeric';
      features[c] = { name: c, type };
    }

    for (const [k, f] of Object.entries(features)) {
      if (f.type === 'numeric') {
        const arr = this.raw.map(r => this.#num(r[k])).filter(Number.isFinite);
        const min = Math.min(...arr), max = Math.max(...arr);
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(1, (arr.length - 1)));
        f.stats = { min, max, mean, std };
      }
    }

    this.schema = { features, target };
  }

  prepareMatrices() {
    this.encoders = {};
    this.featNames = [];
    for (const [k, f] of Object.entries(this.schema.features)) {
      if (f.type === 'numeric') {
        this.featNames.push(k);
      }
    }

    const X = [];
    const y = [];
    for (const r of this.raw) {
      const row = [];
      for (const [k, f] of Object.entries(this.schema.features)) {
        if (f.type === 'numeric') {
          row.push(this.#num(r[k]));
        }
      }
      X.push(row);
      y.push([String(r[this.schema.target])]);
    }

    this.scaler = { type: 'minmax', stats: {} };
    const numericIdx = [];
    let col = 0;
    for (const [k, f] of Object.entries(this.schema.features)) {
      if (f.type === 'numeric') {
        numericIdx.push(col); col += 1;
      }
    }

    for (const c of numericIdx) {
      const colVals = X.map(r => r[c]).filter(Number.isFinite);
      const min = Math.min(...colVals), max = Math.max(...colVals);
      const mean = colVals.reduce((a, b) => a + b, 0) / Math.max(1, colVals.length);
      const std = Math.sqrt(colVals.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(1, (colVals.length - 1)));
      this.scaler.stats[c] = { min, max, mean, std };
    }

    for (let i = 0; i < X.length; i++) {
      for (const c of numericIdx) {
        const st = this.scaler.stats[c]; const v = X[i][c];
        if (!Number.isFinite(v)) { X[i][c] = 0; continue; }
        const d = (st.max - st.min) || 1; X[i][c] = (v - st.min) / d;
      }
    }

    this.X = X;
    this.y = y;
    const idx = [...Array(X.length).keys()];
    this.#shuffle(idx, 2025);
    const nTr = Math.floor(idx.length * 0.8);
    this.idx.train = idx.slice(0, nTr);
    this.idx.test = idx.slice(nTr);
    return { featNames: this.featNames };
  }

  getTrain() { return this.idx.train.map(i => this.X[i]); }
  getTrainY() { return this.idx.train.map(i => this.y[i]); }
  getTest() { return this.idx.test.map(i => this.X[i]); }
  getTestY() { return this.idx.test.map(i => this.y[i]); }

  #shuffle(a, seed = 123) {
    let s = seed;
    const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
  }
}
