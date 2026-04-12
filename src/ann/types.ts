/** Feature order used everywhere: mix + SEM-derived scalars */
export const FEATURE_LABELS = [
  'Cement (kg/m³)',
  'Water (kg/m³)',
  'Copper slag %',
  'Curing days',
  'Porosity %',
  'Crack density (mm/mm²)',
] as const;

export type FeatureRow = [number, number, number, number, number, number];

export interface Normalization {
  mean: number[];
  std: number[];
  yMean: number;
  yStd: number;
}

export interface TrainMetrics {
  rmse: number;
  mae: number;
  r2: number;
  predictions: number[];
  actuals: number[];
}

export interface EpochLog {
  epoch: number;
  loss: number;
  valLoss?: number;
}
