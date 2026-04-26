/**
 * Schema v2 — IS-aware, multi-output.
 *
 * Visible UI inputs (10 raw features). The ANN itself sees these 10 plus 4
 * auto-derived ratios (w/c, c/fa, c/ca, fa/ca) for a total of 14 inputs.
 */
export const FEATURE_LABELS = [
  'Cement (kg/m³)',
  'Water (kg/m³)',
  'Fine aggregate (kg/m³)',
  'Coarse aggregate (kg/m³)',
  'Copper slag %',
  'Curing days',
  'SG cement',
  'SG fine agg',
  'SG coarse agg',
  'Fineness modulus',
] as const;

export const N_RAW_FEATURES = 10;

export const DERIVED_FEATURE_LABELS = [
  'w/c ratio',
  'cement / fine',
  'cement / coarse',
  'fine / coarse',
] as const;

export const N_ANN_FEATURES = N_RAW_FEATURES + DERIVED_FEATURE_LABELS.length;

export type FeatureRow = [
  number, number, number, number, number,
  number, number, number, number, number,
];

/** Sensible defaults used to fill columns missing from legacy CSVs (e.g. UCI). */
export const FEATURE_DEFAULTS: FeatureRow = [
  340, 165, 700, 1150, 0, 28, 3.15, 2.65, 2.7, 2.7,
];

export const OUTPUT_KEYS = ['fck', 'fst', 'ffl', 'ec', 'density'] as const;
export type OutputKey = (typeof OUTPUT_KEYS)[number];

export const OUTPUT_LABELS: Record<OutputKey, string> = {
  fck: 'Compressive strength',
  fst: 'Split tensile',
  ffl: 'Flexural strength',
  ec: 'Modulus of elasticity',
  density: 'Density',
};

export const OUTPUT_UNITS: Record<OutputKey, string> = {
  fck: 'MPa',
  fst: 'MPa',
  ffl: 'MPa',
  ec: 'GPa',
  density: 'kg/m³',
};

export const OUTPUT_SHORT: Record<OutputKey, string> = {
  fck: "f'c",
  fst: 'fst',
  ffl: 'fr',
  ec: 'Ec',
  density: 'ρ',
};

export interface Normalization {
  xMean: number[];
  xStd: number[];
  yMean: number[];
  yStd: number[];
  outputs: OutputKey[];
}

export interface OutputMetrics {
  rmse: number;
  mae: number;
  r2: number;
  predictions: number[];
  actuals: number[];
}

export type TrainMetrics = Partial<Record<OutputKey, OutputMetrics>>;

export interface EpochLog {
  epoch: number;
  loss: number;
  valLoss?: number;
}

/**
 * Live training progress emitted from `trainAnn` once per batch (and once at
 * the start of every epoch with `samplesProcessed = 0`).
 */
export interface BatchProgress {
  /** Rows from the training split processed so far in the current epoch. */
  samplesProcessed: number;
  /** Total rows in the training split (i.e. after the validation slice is removed). */
  trainSize: number;
  /** 1-indexed epoch currently in progress. */
  epoch: number;
  /** Configured total number of epochs (training may stop early). */
  totalEpochs: number;
}

/** Compute the 4 derived ratios appended to raw features before training. */
export function deriveRatios(row: FeatureRow): [number, number, number, number] {
  const [cement, water, fine, coarse] = row;
  const safe = (n: number) => (Math.abs(n) < 1e-3 ? 1e-3 : n);
  return [
    water / safe(cement),
    cement / safe(fine),
    cement / safe(coarse),
    fine / safe(coarse),
  ];
}

/** Convert a raw input row to the full ANN feature vector (length = N_ANN_FEATURES). */
export function annFeatures(row: FeatureRow): number[] {
  const ratios = deriveRatios(row);
  return [...row, ...ratios];
}
