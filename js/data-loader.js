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

  // Загрузка CSV
  async loadCSV(path = 'https://vovanbaklazhan.github.io/UCI-HAR/data/train.csv') {
    this.setStatus('loading data…');
    this.log(`Fetching ${path}`);
    this.raw = [];

    try {
      const res = await fetch(path);  // Используем fetch асинхронно
      if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
      const text = await res.text();  // Получаем текст данных

      // Проверка на пустоту
      if (!text.trim()) {
        throw new Error(`CSV file at ${path} is empty.`);
      }

      const data = this.parseCSV(text);  // Парсим CSV
      if (!data.length) throw new Error(`CSV ${path} has no valid rows.`);
      this.raw = [...this.raw, ...data]; // Объединяем данные
      this.log(`Loaded ${data.length} rows from ${path}`);
    } catch (e) {
      this.log(`Error loading ${path}: ${e.message}`);
      throw e;
    }

    // Логируем первые 5 строк данных для отладки
    console.log('Raw data preview:', this.raw.slice(0, 5)); 

    this.inferSchema();  // Преобразуем схему данных
    this.setStatus('data loaded');
    this.log(`Total rows loaded: ${this.raw.length}`);
  }

  // Парсинг CSV
  parseCSV(text) {
    const [h, ...lines] = text.trim().split(/\r?\n/);
    const headers = h.split(',').map(s => s.trim());

    // Логируем заголовки столбцов
    console.log('Headers:', headers);

    return lines.map((line, idx) => {
      const cells = line.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g).map(v => v.replace(/^"(.*)"$/, '$1').trim());
      
      const o = {};
      headers.forEach((k, i) => {
        o[k] = cells[i] ?? '';
      });

      // Логируем строки данных
      if (idx < 5) { 
        console.log(`Row ${idx}:`, o);
      }

      return o;
    });
  }

  // Преобразование строки в число
  num(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }

  // Инференция схемы
  inferSchema() {
    const target = 'Activity'; // Целевой столбец
    const headers = Object.keys(this.raw[0]);

    // Логируем доступные заголовки
    const cleanedHeaders = headers.map(header => header.trim());
    console.log('Cleaned headers:', cleanedHeaders);

    // Приводим заголовки и целевой столбец к единому регистру
    const cleanedTarget = target.trim().toLowerCase();
    const headerFound = cleanedHeaders.some(header => header.toLowerCase() === cleanedTarget);

    if (!headerFound) {
      throw new Error(`Target "${target}" not found`);
    }

    const features = {};
    const cols = cleanedHeaders.filter(c => c.toLowerCase() !== cleanedTarget); // Исключаем целевой признак

    for (const c of cols) {
      let type = 'numeric';
      features[c] = { name: c, type };
    }

    for (const [k, f] of Object.entries(features)) {
      if (f.type === 'numeric') {
        const arr = this.raw.map(r => this.num(r[k])).filter(Number.isFinite);
        const min = Math.min(...arr), max = Math.max(...arr);
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(1, (arr.length - 1)));
        f.stats = { min, max, mean, std };
      }
    }

    this.schema = { features, target };
  }

  // Подготовка матриц для обучения
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
      for (const [k, f] of Object.entries(this.s
