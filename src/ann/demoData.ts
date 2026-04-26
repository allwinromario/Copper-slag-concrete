import {
  ecFromFck,
  flexuralFromFck,
  splitTensileFromFck,
} from '../lib/isCode';
import {
  OUTPUT_KEYS,
  type FeatureRow,
  type OutputKey,
} from './types';

/**
 * Synthetic strength recipe (no porosity / crack — uses the new schema).
 *
 * Drivers:
 *   • base fck rises with cement and falls with water/cement ratio
 *   • copper slag has a mild peak around 30% replacement
 *   • fineness modulus contributes most when near ~2.7 (Zone II)
 *   • age factor saturates around 28 days
 *
 * IS-anchored secondary outputs are seeded from fck via IS 456 formulas plus
 * Gaussian-ish noise so the targets are correlated but not identical.
 */
export function syntheticStrengthCore(row: FeatureRow): number {
  const [cement, water, , , slag, days, , , , fm] = row;
  const wcm = water / Math.max(50, cement);
  const slagFactor = 1 + 0.05 * (slag / 30) - 0.18 * Math.pow((slag - 30) / 30, 2);
  const fmFactor = 1 - 0.06 * Math.abs(fm - 2.7);
  const ageFactor = 0.55 + 0.45 * Math.min(1, Math.log1p(days) / Math.log1p(28));
  const base = 16 + 0.06 * cement - 70 * wcm;
  return base * slagFactor * fmFactor * ageFactor;
}

export function syntheticStrengthHint(row: FeatureRow): number {
  return Math.max(8, syntheticStrengthCore(row));
}

function densityFromMix(row: FeatureRow): number {
  const [cement, water, fine, coarse, , , sgc, sgf, sgca] = row;
  const Vc = cement / (sgc * 1000);
  const Vw = water / 1000;
  const Vf = fine / (sgf * 1000);
  const Vca = coarse / (sgca * 1000);
  const totalMass = cement + water + fine + coarse;
  const totalVol = Math.max(0.6, Vc + Vw + Vf + Vca);
  return totalMass / totalVol;
}

/** Compute all five synthetic outputs for a single row. */
export function syntheticOutputs(row: FeatureRow): Record<OutputKey, number> {
  const fck = Math.max(8, syntheticStrengthCore(row));
  const fst = splitTensileFromFck(fck) * (0.92 + Math.random() * 0.16);
  const ffl = flexuralFromFck(fck) * (0.95 + Math.random() * 0.18);
  const ec = ecFromFck(fck) * (0.95 + Math.random() * 0.1);
  const density = densityFromMix(row) * (0.985 + Math.random() * 0.03);
  return { fck, fst, ffl, ec, density };
}

/** Plausible bounds for synthetic generation; mirrored in `optimize.ts`. */
const RANGE: [number, number][] = [
  [280, 420],
  [140, 200],
  [620, 820],
  [1050, 1250],
  [0, 50],
  [7, 56],
  [3.05, 3.18],
  [2.55, 2.75],
  [2.6, 2.85],
  [2.4, 3.0],
];

const CURING_DAYS = [7, 14, 28, 56] as const;

export function generateDemoData(n: number = 100): {
  X: FeatureRow[];
  y: Record<OutputKey, number[]>;
} {
  const X: FeatureRow[] = [];
  const y: Record<OutputKey, number[]> = {
    fck: [],
    fst: [],
    ffl: [],
    ec: [],
    density: [],
  };

  for (let i = 0; i < n; i++) {
    const row = [
      RANGE[0][0] + Math.random() * (RANGE[0][1] - RANGE[0][0]),
      RANGE[1][0] + Math.random() * (RANGE[1][1] - RANGE[1][0]),
      RANGE[2][0] + Math.random() * (RANGE[2][1] - RANGE[2][0]),
      RANGE[3][0] + Math.random() * (RANGE[3][1] - RANGE[3][0]),
      Math.random() * 50,
      CURING_DAYS[Math.floor(Math.random() * CURING_DAYS.length)],
      RANGE[6][0] + Math.random() * (RANGE[6][1] - RANGE[6][0]),
      RANGE[7][0] + Math.random() * (RANGE[7][1] - RANGE[7][0]),
      RANGE[8][0] + Math.random() * (RANGE[8][1] - RANGE[8][0]),
      RANGE[9][0] + Math.random() * (RANGE[9][1] - RANGE[9][0]),
    ] as FeatureRow;

    const out = syntheticOutputs(row);
    X.push(row);
    for (const k of OUTPUT_KEYS) y[k].push(round(out[k], k === 'density' ? 0 : 2));
  }
  return { X, y };
}

function round(v: number, dp: number): number {
  const f = Math.pow(10, dp);
  return Math.round(v * f) / f;
}

export const DEMO_CSV_HEADER =
  'Cement,Water,Fine aggregate,Coarse aggregate,Copper slag %,Curing days,SG cement,SG fine,SG coarse,Fineness modulus,Compressive strength,Split tensile,Flexural strength,Modulus of elasticity,Density';

function fmt(n: number, dp: number): string {
  return (Math.round(n * Math.pow(10, dp)) / Math.pow(10, dp)).toFixed(dp);
}

export function buildDemoCsv(n: number = 12): string {
  const { X, y } = generateDemoData(n);
  const rows = X.map(
    (r, i) =>
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
        fmt(y.fck[i], 2),
        fmt(y.fst[i], 2),
        fmt(y.ffl[i], 2),
        fmt(y.ec[i], 2),
        fmt(y.density[i], 0),
      ].join(',')
  );
  return [DEMO_CSV_HEADER, ...rows].join('\n');
}

/** Small fixed demo so the app loads with stable rows on first paint. */
export const DEMO_CSV = `${DEMO_CSV_HEADER}
350,165,720,1180,0,28,3.15,2.65,2.72,2.7,38.4,4.2,4.6,30.5,2380
360,160,710,1190,10,28,3.15,2.62,2.71,2.68,40.1,4.4,4.8,31.2,2395
340,170,730,1170,20,28,3.15,2.6,2.7,2.65,37.2,4.1,4.5,30.0,2370
355,158,705,1195,30,28,3.15,2.58,2.72,2.7,38.9,4.3,4.7,30.7,2388
345,172,715,1175,40,28,3.15,2.56,2.71,2.62,33.5,3.9,4.2,28.5,2365
365,155,700,1205,5,56,3.15,2.65,2.73,2.72,44.6,4.7,5.1,33.1,2400
335,168,720,1180,15,14,3.15,2.63,2.7,2.68,28.9,3.5,3.8,26.6,2375
350,162,715,1185,25,28,3.15,2.61,2.71,2.69,36.4,4.0,4.5,29.6,2382
325,175,730,1160,35,7,3.15,2.59,2.69,2.6,19.7,3.0,3.2,21.9,2355
370,150,695,1210,0,56,3.15,2.66,2.74,2.72,46.8,4.8,5.3,33.8,2410
`;
