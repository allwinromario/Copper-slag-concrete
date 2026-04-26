import { generateDemoData, syntheticStrengthHint } from './demoData';
import {
  isCompliant,
  type Exposure,
  EXPOSURE_KEYS,
} from '../lib/isCode';
import {
  type FeatureRow,
  type Normalization,
  type OutputKey,
} from './types';
import { predictStrength } from './trainAnn';
import type * as tfTypes from '@tensorflow/tfjs';

export const OPTIMAL_TRAINING = {
  syntheticRowCount: 360,
  epochs: 500,
  learningRate: 0.008,
  validationFraction: 0.18,
} as const;

const CURING_DAYS = [7, 14, 28, 56] as const;

/** [min, max] per raw feature index. Days is enumerated separately. */
const BOUNDS: { min: number; max: number; discreteDays?: boolean }[] = [
  { min: 280, max: 420 },
  { min: 140, max: 200 },
  { min: 620, max: 820 },
  { min: 1050, max: 1250 },
  { min: 0, max: 50 },
  { min: 7, max: 56, discreteDays: true },
  { min: 3.05, max: 3.18 },
  { min: 2.55, max: 2.75 },
  { min: 2.6, max: 2.85 },
  { min: 2.4, max: 3.0 },
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
    BOUNDS[3].min + Math.random() * (BOUNDS[3].max - BOUNDS[3].min),
    BOUNDS[4].min + Math.random() * (BOUNDS[4].max - BOUNDS[4].min),
    days,
    BOUNDS[6].min + Math.random() * (BOUNDS[6].max - BOUNDS[6].min),
    BOUNDS[7].min + Math.random() * (BOUNDS[7].max - BOUNDS[7].min),
    BOUNDS[8].min + Math.random() * (BOUNDS[8].max - BOUNDS[8].min),
    BOUNDS[9].min + Math.random() * (BOUNDS[9].max - BOUNDS[9].min),
  ] as FeatureRow;
}

function snapRow(r: FeatureRow): FeatureRow {
  const nearestDay = CURING_DAYS.reduce((best, d) =>
    Math.abs(d - r[5]) < Math.abs(best - r[5]) ? d : best
  );
  return [
    Math.round(r[0] * 10) / 10,
    Math.round(r[1] * 10) / 10,
    Math.round(r[2] * 10) / 10,
    Math.round(r[3] * 10) / 10,
    Math.round(r[4] * 10) / 10,
    nearestDay,
    Math.round(r[6] * 100) / 100,
    Math.round(r[7] * 100) / 100,
    Math.round(r[8] * 100) / 100,
    Math.round(r[9] * 100) / 100,
  ] as FeatureRow;
}

function rowToMix(r: FeatureRow) {
  return {
    cement: r[0],
    water: r[1],
    fineAgg: r[2],
    coarseAgg: r[3],
    copperSlagPct: r[4],
    fm: r[9],
  };
}

function yieldToMain(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

const SEARCH_CHUNK = 280;

async function searchMaxByScore(
  score: (row: FeatureRow) => number,
  exposure: Exposure | null,
  randomTrials = 8000,
  refineSteps = 600
): Promise<FeatureRow> {
  const accept = (row: FeatureRow): boolean =>
    !exposure || isCompliant(rowToMix(row), exposure);

  let best = snapRow(randomFeatureRow());
  let attempts = 0;
  while (!accept(best) && attempts++ < 500) best = snapRow(randomFeatureRow());
  let bestY = score(best);

  let t = 0;
  while (t < randomTrials) {
    const end = Math.min(t + SEARCH_CHUNK, randomTrials);
    for (; t < end; t++) {
      const row = snapRow(randomFeatureRow());
      if (!accept(row)) continue;
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
      const dim = Math.floor(Math.random() * 10);
      const b = BOUNDS[dim];
      if (b.discreteDays) {
        cand[5] = CURING_DAYS[Math.floor(Math.random() * CURING_DAYS.length)];
      } else {
        const span = b.max - b.min;
        const step = (Math.random() - 0.5) * span * 0.08;
        cand[dim] = clamp(cand[dim] + step, b.min, b.max);
      }
      const snapped = snapRow(cand);
      if (!accept(snapped)) continue;
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

export async function searchMaxStrengthInputs(
  model: tfTypes.Sequential,
  norm: Normalization,
  exposure: Exposure | null = null,
  randomTrials = 8000,
  refineSteps = 600
): Promise<FeatureRow> {
  return searchMaxByScore(
    (row) => predictStrength(model, norm, row),
    exposure,
    randomTrials,
    refineSteps
  );
}

export async function searchMaxStrengthSynthetic(
  exposure: Exposure | null = null,
  randomTrials = 10000,
  refineSteps = 800
): Promise<FeatureRow> {
  return searchMaxByScore(syntheticStrengthHint, exposure, randomTrials, refineSteps);
}

export function buildOptimizedSyntheticCsv(): string {
  const n = Math.min(500, Math.max(20, OPTIMAL_TRAINING.syntheticRowCount));
  const { X, y } = generateDemoData(n);
  const header =
    'Cement,Water,Fine aggregate,Coarse aggregate,Copper slag %,Curing days,SG cement,SG fine,SG coarse,Fineness modulus,Compressive strength,Split tensile,Flexural strength,Modulus of elasticity,Density';
  const fmt = (v: number, dp: number) => v.toFixed(dp);
  const keys: OutputKey[] = ['fck', 'fst', 'ffl', 'ec', 'density'];
  const rows = X.map((r, i) =>
    [
      fmt(r[0], 1),
      fmt(r[1], 1),
      fmt(r[2], 1),
      fmt(r[3], 1),
      fmt(r[4], 1),
      r[5],
      fmt(r[6], 2),
      fmt(r[7], 2),
      fmt(r[8], 2),
      fmt(r[9], 2),
      ...keys.map((k) => fmt(y[k][i], k === 'density' ? 0 : 2)),
    ].join(',')
  );
  return [header, ...rows].join('\n');
}

export { EXPOSURE_KEYS };
