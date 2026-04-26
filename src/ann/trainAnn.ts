import * as tf from '@tensorflow/tfjs';
import {
  N_ANN_FEATURES,
  OUTPUT_KEYS,
  annFeatures,
  type BatchProgress,
  type EpochLog,
  type FeatureRow,
  type Normalization,
  type OutputKey,
  type OutputMetrics,
  type TrainMetrics,
} from './types';

function shuffleIndices(n: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = idx[i];
    idx[i] = idx[j];
    idx[j] = tmp;
  }
  return idx;
}

function computeStats(values: number[]): { mean: number; std: number } {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return { mean: 0, std: 1 };
  const mean = finite.reduce((a, b) => a + b, 0) / finite.length;
  let v = 0;
  for (const x of finite) v += (x - mean) * (x - mean);
  const std = Math.sqrt(v / finite.length) || 1;
  return { mean, std };
}

function buildAnnMatrix(rows: FeatureRow[]): number[][] {
  return rows.map((r) => annFeatures(r));
}

function tensorNorm2d(matrix: number[][], xMean: number[], xStd: number[]): tf.Tensor2D {
  const out = matrix.map((row) =>
    row.map((v, f) => (v - xMean[f]) / xStd[f])
  );
  return tf.tensor2d(out);
}

export const MODEL_ARCHITECTURE = [
  { layer: 'Input', shape: '(14)', activation: '—', notes: '10 raw + 4 derived ratios' },
  { layer: 'Dense', units: 64, activation: 'ReLU', notes: 'Hidden 1' },
  { layer: 'Dense', units: 48, activation: 'ReLU', notes: 'Hidden 2' },
  { layer: 'Dense', units: 32, activation: 'ReLU', notes: 'Hidden 3' },
  { layer: 'Dense', units: 'k', activation: 'Linear', notes: 'k = available outputs (1–5), normalized' },
] as const;

const EARLY_STOP_PATIENCE = 36;
const EARLY_STOP_MIN_DELTA = 1e-6;

export interface TrainingDataset {
  X: FeatureRow[];
  y: Partial<Record<OutputKey, number[]>>;
  availableOutputs: OutputKey[];
}

export interface TrainOptions {
  epochs: number;
  learningRate: number;
  validationFraction: number;
  onEpoch: (log: EpochLog) => void;
  /** Optional per-batch progress (rows processed within the current epoch). */
  onBatch?: (progress: BatchProgress) => void;
}

export interface TrainedBundle {
  model: tf.Sequential;
  norm: Normalization;
  testMetrics: TrainMetrics;
  outputs: OutputKey[];
  rowsUsed: number;
  rowsDroppedForNaN: number;
}

export async function trainAnn(
  dataset: TrainingDataset,
  options: TrainOptions
): Promise<TrainedBundle> {
  const outputs = dataset.availableOutputs.length
    ? dataset.availableOutputs
    : (['fck'] as OutputKey[]);

  const totalRows = dataset.X.length;
  const keepIdx: number[] = [];
  for (let i = 0; i < totalRows; i++) {
    let ok = true;
    for (const k of outputs) {
      const arr = dataset.y[k];
      if (!arr || !Number.isFinite(arr[i])) {
        ok = false;
        break;
      }
    }
    if (ok) keepIdx.push(i);
  }

  if (keepIdx.length < 4) {
    throw new Error(
      'Not enough rows with values for every available output. Try a CSV with fewer outputs or fill in the missing cells.'
    );
  }

  const X = keepIdx.map((i) => dataset.X[i]);
  const yByKey: Record<string, number[]> = {};
  for (const k of outputs) yByKey[k] = keepIdx.map((i) => dataset.y[k]![i]);

  const order = shuffleIndices(X.length);
  const Xs = order.map((i) => X[i]);
  const Ys: Record<string, number[]> = {};
  for (const k of outputs) Ys[k] = order.map((i) => yByKey[k][i]);

  const nVal = Math.max(1, Math.floor(Xs.length * options.validationFraction));
  const XTrain = Xs.slice(nVal);
  const XVal = Xs.slice(0, nVal);
  const YTrain: Record<string, number[]> = {};
  const YVal: Record<string, number[]> = {};
  for (const k of outputs) {
    YTrain[k] = Ys[k].slice(nVal);
    YVal[k] = Ys[k].slice(0, nVal);
  }

  const xTrainMatrix = buildAnnMatrix(XTrain);
  const xValMatrix = buildAnnMatrix(XVal);

  const xMean = Array(N_ANN_FEATURES).fill(0);
  const xStd = Array(N_ANN_FEATURES).fill(1);
  for (let f = 0; f < N_ANN_FEATURES; f++) {
    const col = xTrainMatrix.map((r) => r[f]);
    const s = computeStats(col);
    xMean[f] = s.mean;
    xStd[f] = s.std || 1;
  }

  const yMean: number[] = [];
  const yStd: number[] = [];
  for (const k of outputs) {
    const s = computeStats(YTrain[k]);
    yMean.push(s.mean);
    yStd.push(s.std || 1);
  }

  const xTrainT = tensorNorm2d(xTrainMatrix, xMean, xStd);
  const xValT = tensorNorm2d(xValMatrix, xMean, xStd);
  const yTrainT = tf.tensor2d(
    YTrain[outputs[0]].map((_, i) =>
      outputs.map((k, kIdx) => (YTrain[k][i] - yMean[kIdx]) / yStd[kIdx])
    )
  );
  const yValT = tf.tensor2d(
    YVal[outputs[0]].map((_, i) =>
      outputs.map((k, kIdx) => (YVal[k][i] - yMean[kIdx]) / yStd[kIdx])
    )
  );

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 64,
      activation: 'relu',
      inputShape: [N_ANN_FEATURES],
    })
  );
  model.add(tf.layers.dense({ units: 48, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: outputs.length, activation: 'linear' }));

  model.compile({
    optimizer: tf.train.adam(options.learningRate),
    loss: 'meanSquaredError',
  });

  let bestVal = Infinity;
  const earlyBest: { weights: tf.Tensor[] | null } = { weights: null };
  let patienceLeft = EARLY_STOP_PATIENCE;

  const trainSize = XTrain.length;
  const batchSize = Math.min(64, Math.max(8, Math.floor(trainSize / 5)));
  let currentEpoch = 1;

  options.onBatch?.({
    samplesProcessed: 0,
    trainSize,
    epoch: 1,
    totalEpochs: options.epochs,
  });

  await model.fit(xTrainT, yTrainT, {
    epochs: options.epochs,
    batchSize,
    shuffle: true,
    validationData: [xValT, yValT],
    callbacks: {
      onEpochBegin: async (epoch) => {
        currentEpoch = epoch + 1;
        options.onBatch?.({
          samplesProcessed: 0,
          trainSize,
          epoch: currentEpoch,
          totalEpochs: options.epochs,
        });
      },
      onBatchEnd: async (batch) => {
        if (!options.onBatch) return;
        const samples = Math.min((batch + 1) * batchSize, trainSize);
        options.onBatch({
          samplesProcessed: samples,
          trainSize,
          epoch: currentEpoch,
          totalEpochs: options.epochs,
        });
      },
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
          if (patienceLeft <= 0) model.stopTraining = true;
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

  const testMetrics: TrainMetrics = {};
  for (let kIdx = 0; kIdx < outputs.length; kIdx++) {
    const k = outputs[kIdx];
    const predictions = predArr.map((row) => row[kIdx] * yStd[kIdx] + yMean[kIdx]);
    const actuals = YVal[k];
    testMetrics[k] = computeMetrics(predictions, actuals);
  }

  xTrainT.dispose();
  yTrainT.dispose();
  xValT.dispose();
  yValT.dispose();

  return {
    model,
    norm: { xMean, xStd, yMean, yStd, outputs },
    testMetrics,
    outputs,
    rowsUsed: X.length,
    rowsDroppedForNaN: totalRows - X.length,
  };
}

function computeMetrics(predictions: number[], actuals: number[]): OutputMetrics {
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

/** Predict all trained outputs for one raw input row. */
export function predictAll(
  model: tf.Sequential,
  norm: Normalization,
  row: FeatureRow
): Partial<Record<OutputKey, number>> {
  const ann = annFeatures(row);
  const xRow = ann.map((v, f) => (v - norm.xMean[f]) / norm.xStd[f]);
  const x = tf.tensor2d([xRow]);
  const y = model.predict(x) as tf.Tensor;
  const arr = y.dataSync();
  x.dispose();
  y.dispose();
  const out: Partial<Record<OutputKey, number>> = {};
  for (let i = 0; i < norm.outputs.length; i++) {
    const k = norm.outputs[i];
    out[k] = arr[i] * norm.yStd[i] + norm.yMean[i];
  }
  return out;
}

/** Convenience: predict only fck (used by the optimizer). */
export function predictStrength(
  model: tf.Sequential,
  norm: Normalization,
  row: FeatureRow
): number {
  const out = predictAll(model, norm, row);
  if (out.fck != null) return out.fck;
  for (const k of OUTPUT_KEYS) {
    if (out[k] != null) return out[k]!;
  }
  return 0;
}
