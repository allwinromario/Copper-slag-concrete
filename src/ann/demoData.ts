import type { FeatureRow } from './types';

/** Unclamped strength formula (same as pre-noise `generateDemoData`). */
export function syntheticStrengthCore(row: FeatureRow): number {
  const [cement, water, slag, days, porosity, crack] = row;
  const wcm = water / (cement + 0.01);
  return (
    15 +
    0.045 * cement -
    28 * wcm +
    0.12 * slag -
    0.35 * slag * (slag / 100) +
    0.22 * Math.log1p(days) -
    0.55 * porosity -
    0.4 * crack
  );
}

/**
 * Noise-free score for a mix (floor at 8). Used to suggest strong inputs before an ANN exists.
 */
export function syntheticStrengthHint(row: FeatureRow): number {
  return Math.max(8, syntheticStrengthCore(row));
}

/** Synthetic demo data: plausible ranges + nonlinear-ish strength */
export function generateDemoData(n: number = 80): { X: FeatureRow[]; y: number[] } {
  const X: FeatureRow[] = [];
  const y: number[] = [];

  for (let i = 0; i < n; i++) {
    const cement = 280 + Math.random() * 120;
    const water = 140 + Math.random() * 60;
    const slag = Math.random() * 50;
    const days = [7, 14, 28, 56][Math.floor(Math.random() * 4)];
    const porosity = 4 + Math.random() * 12;
    const crack = Math.random() * 8;

    const row: FeatureRow = [cement, water, slag, days, porosity, crack];
    const noise = (Math.random() - 0.5) * 3;
    const strength = Math.max(8, syntheticStrengthCore(row) + noise);

    X.push(row);
    y.push(Math.round(strength * 10) / 10);
  }

  return { X, y };
}

export const DEMO_CSV = `Cement,Water,Copper slag %,Curing days,Porosity %,Crack density,Compressive strength
320,165,20,28,7.2,2.1,38.4
340,160,10,28,6.1,1.4,42.1
300,170,30,28,9.5,3.2,31.8
350,155,0,28,5.8,1.1,44.6
310,168,25,14,8.4,2.8,28.9
330,162,15,56,6.5,1.6,45.2
295,175,40,28,11.2,4.1,24.3
360,150,5,28,5.2,0.9,46.8
315,172,35,7,10.1,3.5,19.7
325,166,22,28,7.8,2.4,35.6
`;
