import type { FeatureRow } from '../ann/types';

const STORAGE_KEY = 'ann.inferenceFeatureRow.v1';

export function loadStoredFeatureInputs(fallback: FeatureRow): FeatureRow {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed) || parsed.length !== 6) return fallback;
    const nums = parsed.map((x) => Number(x));
    if (!nums.every((x) => Number.isFinite(x))) return fallback;
    return nums as FeatureRow;
  } catch {
    return fallback;
  }
}

export function persistFeatureInputs(row: FeatureRow): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(row));
  } catch {
    // private mode / quota
  }
}
