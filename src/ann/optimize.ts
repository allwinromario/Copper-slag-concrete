import { generateDemoData, syntheticStrengthHint } from './demoData';
import type { FeatureRow, Normalization } from './types';
import { predictStrength } from './trainAnn';
import type * as tfTypes from '@tensorflow/tfjs';

/** Tuned for synthetic data + Adam: more rows, longer cap (early stopping trims), moderate LR. */
export const OPTIMAL_TRAINING = {
  syntheticRowCount: 360,
  epochs: 450,
  learningRate: 0.009,
  validationFraction: 0.18,
} as const;

const CURING_DAYS = [7, 14, 28, 56] as const;

/** Plausible bounds aligned with `generateDemoData` (slightly widened for search). */
const BOUNDS: { min: number; max: number; discreteDays?: boolean }[] = [
  { min: 275, max: 405 },
  { min: 132, max: 205 },
  { min: 0, max: 50 },
  { min: 7, max: 56, discreteDays: true },
  { min: 3.5, max: 17 },
  { min: 0, max: 8.5 },
];

function clamp(v: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, v));
}

function randomFeatureRow(): FeatureRow {
  const days = CURING_DAYS[Math.floor(Math.random() * CURING_DAYS.length)];
  return [
    BOUNDS[0].min + Math.random() * (BOUNDS[0].max - BOUNDS[0].min),
    BOUNDS[1].min + Math.random() * (BOUNDS[1].max - BOUNDS[1].min),
    BOUNDS[2].min + Math.random() * (BOUNDS[2].max - BOUNDS[2].min),
    days,
    BOUNDS[4].min + Math.random() * (BOUNDS[4].max - BOUNDS[4].min),
    BOUNDS[5].min + Math.random() * (BOUNDS[5].max - BOUNDS[5].min),
  ] as FeatureRow;
}

function snapRow(r: FeatureRow): FeatureRow {
  const nearestDay = CURING_DAYS.reduce((best, d) =>
    Math.abs(d - r[3]) < Math.abs(best - r[3]) ? d : best
  );
  return [
    Math.round(r[0] * 10) / 10,
    Math.round(r[1] * 10) / 10,
    Math.round(r[2] * 10) / 10,
    nearestDay,
    Math.round(r[4] * 100) / 100,
    Math.round(r[5] * 100) / 100,
  ] as FeatureRow;
}

/** Lets the browser paint and handle input between heavy TF.js batches. */
function yieldToMain(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

const SEARCH_CHUNK = 280;

async function searchMaxByScore(
  score: (row: FeatureRow) => number,
  randomTrials = 10_000,
  refineSteps = 600
): Promise<FeatureRow> {
  let best = snapRow(randomFeatureRow());
  let bestY = score(best);

  let t = 0;
  while (t < randomTrials) {
    const end = Math.min(t + SEARCH_CHUNK, randomTrials);
    for (; t < end; t++) {
      const row = snapRow(randomFeatureRow());
      const y = score(row);
      if (y > bestY) {
        bestY = y;
        best = row;
      }
    }
    await yieldToMain();
  }

  let cur = [...best] as FeatureRow;
  let curY = bestY;
  let s = 0;
  while (s < refineSteps) {
    const end = Math.min(s + SEARCH_CHUNK, refineSteps);
    for (; s < end; s++) {
      const cand = [...cur] as FeatureRow;
      const dim = Math.floor(Math.random() * 6);
      const b = BOUNDS[dim];
      if (b.discreteDays) {
        cand[3] = CURING_DAYS[Math.floor(Math.random() * CURING_DAYS.length)];
      } else {
        const span = b.max - b.min;
        const step = (Math.random() - 0.5) * span * 0.08;
        cand[dim] = clamp(cand[dim] + step, b.min, b.max);
      }
      const snapped = snapRow(cand);
      const y = score(snapped);
      if (y > curY) {
        curY = y;
        cur = snapped;
      }
    }
    await yieldToMain();
  }

  return cur;
}

/**
 * Stochastic search for inputs that maximize predicted strength (given trained model).
 * Chunked + async so the UI thread is not frozen for thousands of forward passes.
 */
export async function searchMaxStrengthInputs(
  model: tfTypes.Sequential,
  norm: Normalization,
  randomTrials = 10_000,
  refineSteps = 600
): Promise<FeatureRow> {
  return searchMaxByScore((row) => predictStrength(model, norm, row), randomTrials, refineSteps);
}

/**
 * Before an ANN exists: maximize the same closed-form demo heuristic as the synthetic dataset.
 */
export async function searchMaxStrengthSynthetic(
  randomTrials = 12_000,
  refineSteps = 800
): Promise<FeatureRow> {
  return searchMaxByScore(syntheticStrengthHint, randomTrials, refineSteps);
}

/** Build CSV text using optimal row count (for data-section Optimize). */
export function buildOptimizedSyntheticCsv(): string {
  const n = Math.min(500, Math.max(20, OPTIMAL_TRAINING.syntheticRowCount));
  const { X, y } = generateDemoData(n);
  const header =
    'Cement,Water,Copper slag %,Curing days,Porosity %,Crack density,Compressive strength';
  const rows = X.map(
    (r, i) =>
      `${r[0].toFixed(1)},${r[1].toFixed(1)},${r[2].toFixed(1)},${r[3]},${r[4].toFixed(2)},${r[5].toFixed(2)},${y[i]}`
  );
  return [header, ...rows].join('\n');
}
