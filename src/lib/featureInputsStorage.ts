import { FEATURE_DEFAULTS, N_RAW_FEATURES, type FeatureRow } from '../ann/types';

/**
 * Bumped to v2 with the new 10-feature schema. Old v1 keys are ignored
 * automatically because the length and key name no longer match.
 */
const STORAGE_KEY = 'ann.inferenceFeatureRow.v2';

export function loadStoredFeatureInputs(fallback: FeatureRow): FeatureRow {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed) || parsed.length !== N_RAW_FEATURES) return fallback;
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

export const FEATURE_INPUT_FALLBACK: FeatureRow = [...FEATURE_DEFAULTS];
