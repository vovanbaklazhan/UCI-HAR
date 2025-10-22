export function buildModel(arch, inputDim, learningRate) {
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

export async function fitModel(model, X, Y, epochs, batchSize, logFn) {
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

export function predictOne(model, x) {
    const xs = tf.tensor2d([x]);
    const pred = model.predict(xs);
    const result = pred.dataSync()[0];
    xs.dispose();
    pred.dispose();
    return result;
}

export function evaluateAccuracy(model, X, Y, threshold = 0.5) {
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
