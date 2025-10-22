export function buildModel(kind, inputLen, lr = 0.001) {
  const m = tf.sequential();
  if (kind === 'cnn1d') {
    m.add(tf.layers.reshape({ targetShape: [inputLen, 1], inputShape: [inputLen] }));
    m.add(tf.layers.conv1d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
    m.add(tf.layers.globalAveragePooling1d());
    m.add(tf.layers.dense({ units: 6, activation: 'softmax' }));  // 6 классов активности
  } else if (kind === 'mlp') {
    m.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [inputLen] }));
    m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    m.add(tf.layers.dense({ units: 6, activation: 'softmax' }));  // 6 классов активности
  } else {
    throw new Error('Unknown model kind');
  }

  m.compile({ optimizer: tf.train.adam(lr), loss: 'sparseCategoricalCrossentropy', metrics: ['mae'] });
  return m;
}

export async function fitModel(model, Xtr, ytr, epochs = 10, batchSize = 256, logFn) {
  const xt = tf.tensor2d(Xtr);
  const yt = tf.tensor2d(ytr);
  const h = await model.fit(xt, yt, {
    epochs, batchSize, validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (ep, logs) => {
        const loss = logs.loss?.toFixed(6);
        const vloss = (logs.val_loss ?? 0).toFixed(6);
        const mae = (logs.mae ?? 0).toFixed(6);
        const vmae = (logs.val_mae ?? 0).toFixed(6);
        logFn?.(`ep ${ep + 1}/${epochs} loss=${loss} val_loss=${vloss} mae=${mae} val_mae=${vmae}`);
      }
    }
  });
  xt.dispose();
  yt.dispose();
  return h;
}

export function predictOne(model, xRow) {
  const xt = tf.tensor2d([xRow]);
  const y = model.predict(xt);
  const v = y.arraySync()[0];
  xt.dispose();
  y.dispose?.();
  return v;
}

export function evaluateAccuracy(model, X, y, thr = 0.5) {
  const xt = tf.tensor2d(X);
  const yp = model.predict(xt);
  const arr = yp.arraySync().map(a => a.indexOf(Math.max(...a)));  // Преобразуем предсказания в индексы
  xt.dispose();
  yp.dispose?.();
  const yTrue = y.map(a => a[0]);
  let correct = 0;
  for (let i = 0; i < arr.length; i++) {
    const p = arr[i];
    const t = yTrue[i];
    if (p === t) correct++;
  }
  return correct / Math.max(1, arr.length);
}
