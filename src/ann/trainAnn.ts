import * as tf from '@tensorflow/tfjs';
import type { EpochLog, FeatureRow, Normalization, TrainMetrics } from './types';

const N_FEATURES = 6;

function shufflePairs(X: FeatureRow[], y: number[]): void {
  for (let i = X.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmpX = X[i];
    X[i] = X[j];
    X[j] = tmpX;
    const tmpY = y[i];
    y[i] = y[j];
    y[j] = tmpY;
  }
}

function computeNorm(X: FeatureRow[], y: number[]): Normalization {
  const mean = Array(N_FEATURES).fill(0);
  const std = Array(N_FEATURES).fill(0);
  for (let f = 0; f < N_FEATURES; f++) {
    let s = 0;
    for (let i = 0; i < X.length; i++) s += X[i][f];
    mean[f] = s / X.length;
  }
  for (let f = 0; f < N_FEATURES; f++) {
    let v = 0;
    for (let i = 0; i < X.length; i++) {
      const d = X[i][f] - mean[f];
      v += d * d;
    }
    std[f] = Math.sqrt(v / X.length) || 1;
  }
  const yMean = y.reduce((a, b) => a + b, 0) / y.length;
  const yVar =
    y.reduce((s, v) => s + (v - yMean) * (v - yMean), 0) / y.length;
  const yStd = Math.sqrt(yVar) || 1;
  return { mean, std, yMean, yStd };
}

function tensorNorm2d(X: FeatureRow[], norm: Normalization): tf.Tensor2D {
  const rows = X.map((row) =>
    row.map((v, f) => (v - norm.mean[f]) / norm.std[f])
  );
  return tf.tensor2d(rows);
}

export const MODEL_ARCHITECTURE = [
  { layer: 'Input', shape: '(6)', activation: '—', notes: 'Normalized features' },
  { layer: 'Dense', units: 56, activation: 'ReLU', notes: 'Hidden 1' },
  { layer: 'Dense', units: 40, activation: 'ReLU', notes: 'Hidden 2' },
  { layer: 'Dense', units: 24, activation: 'ReLU', notes: 'Hidden 3' },
  { layer: 'Dense', units: 1, activation: 'Linear', notes: 'Compressive strength (normalized target)' },
] as const;

const EARLY_STOP_PATIENCE = 32;
const EARLY_STOP_MIN_DELTA = 1e-6;

export interface TrainOptions {
  epochs: number;
  learningRate: number;
  validationFraction: number;
  onEpoch: (log: EpochLog) => void;
}

export interface TrainedBundle {
  model: tf.Sequential;
  norm: Normalization;
  testMetrics: TrainMetrics;
}

export async function trainAnn(
  X: FeatureRow[],
  y: number[],
  options: TrainOptions
): Promise<TrainedBundle> {
  const Xc = X.map((r) => [...r] as FeatureRow);
  const yc = [...y];
  shufflePairs(Xc, yc);

  const nVal = Math.max(
    1,
    Math.floor(Xc.length * options.validationFraction)
  );
  const XTrain = Xc.slice(nVal);
  const yTrain = yc.slice(nVal);
  const XVal = Xc.slice(0, nVal);
  const yVal = yc.slice(0, nVal);

  const norm = computeNorm(XTrain, yTrain);

  const xTrainT = tensorNorm2d(XTrain, norm);
  const yTrainT = tf.tensor2d(yTrain.map((v) => [(v - norm.yMean) / norm.yStd]));
  const xValT = tensorNorm2d(XVal, norm);
  const yValT = tf.tensor2d(yVal.map((v) => [(v - norm.yMean) / norm.yStd]));

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 56,
      activation: 'relu',
      inputShape: [N_FEATURES],
    })
  );
  model.add(tf.layers.dense({ units: 40, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

  model.compile({
    optimizer: tf.train.adam(options.learningRate),
    loss: 'meanSquaredError',
  });

  let bestVal = Infinity;
  const earlyBest: { weights: tf.Tensor[] | null } = { weights: null };
  let patienceLeft = EARLY_STOP_PATIENCE;

  await model.fit(xTrainT, yTrainT, {
    epochs: options.epochs,
    batchSize: Math.min(64, Math.max(8, Math.floor(XTrain.length / 5))),
    shuffle: true,
    validationData: [xValT, yValT],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const vl = logs?.val_loss;
        options.onEpoch({
          epoch: epoch + 1,
          loss: logs?.loss ?? NaN,
          valLoss: vl,
        });
        if (vl == null || !Number.isFinite(vl)) return;
        if (vl < bestVal - EARLY_STOP_MIN_DELTA) {
          bestVal = vl;
          patienceLeft = EARLY_STOP_PATIENCE;
          if (earlyBest.weights) earlyBest.weights.forEach((t) => t.dispose());
          earlyBest.weights = model.getWeights().map((w) => tf.clone(w));
        } else {
          patienceLeft -= 1;
          if (patienceLeft <= 0) {
            model.stopTraining = true;
          }
        }
      },
    },
  });

  if (earlyBest.weights !== null) {
    model.setWeights(earlyBest.weights);
    earlyBest.weights.forEach((t) => t.dispose());
    earlyBest.weights = null;
  }

  const predNorm = model.predict(xValT) as tf.Tensor;
  const predArr = (await predNorm.array()) as number[][];
  predNorm.dispose();

  const predictions = predArr.map((row) => row[0] * norm.yStd + norm.yMean);
  const actuals = [...yVal];

  xTrainT.dispose();
  yTrainT.dispose();
  xValT.dispose();
  yValT.dispose();

  const testMetrics = computeMetrics(predictions, actuals);

  return { model, norm, testMetrics };
}

function computeMetrics(predictions: number[], actuals: number[]): TrainMetrics {
  const n = predictions.length;
  let sse = 0;
  let sae = 0;
  for (let i = 0; i < n; i++) {
    const e = predictions[i] - actuals[i];
    sse += e * e;
    sae += Math.abs(e);
  }
  const rmse = Math.sqrt(sse / n);
  const mae = sae / n;

  const meanY = actuals.reduce((a, b) => a + b, 0) / n;
  let ssTot = 0;
  for (const v of actuals) ssTot += (v - meanY) * (v - meanY);
  const r2 = ssTot > 0 ? 1 - sse / ssTot : 0;

  return { rmse, mae, r2, predictions, actuals };
}

export function predictStrength(
  model: tf.Sequential,
  norm: Normalization,
  row: FeatureRow
): number {
  const x = tensorNorm2d([row], norm);
  const y = model.predict(x) as tf.Tensor;
  const arr = y.dataSync();
  x.dispose();
  y.dispose();
  return arr[0] * norm.yStd + norm.yMean;
}
