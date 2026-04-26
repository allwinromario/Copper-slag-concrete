/**
 * Indian Standards reference helpers.
 *
 * These are *guardrails and reference formulas*, not part of the ANN's
 * learned mapping. They are used to validate inputs/outputs and to constrain
 * the optimizer's search space.
 *
 * Sources (clauses cited inline):
 *   - IS 456:2000  — Plain and Reinforced Concrete (Code of Practice)
 *   - IS 10262:2019 — Concrete Mix Proportioning (Guidelines)
 *   - IS 383:2016  — Coarse and Fine Aggregate for Concrete
 *   - IS 455:2015  — Portland Slag Cement (referenced for slag replacement)
 */

export type Exposure = 'mild' | 'moderate' | 'severe' | 'very_severe' | 'extreme';

export const EXPOSURE_KEYS: Exposure[] = [
  'mild',
  'moderate',
  'severe',
  'very_severe',
  'extreme',
];

export const EXPOSURE_LABELS: Record<Exposure, string> = {
  mild: 'Mild',
  moderate: 'Moderate',
  severe: 'Severe',
  very_severe: 'Very Severe',
  extreme: 'Extreme',
};

/** IS 10262:2019 Table 1 — assumed standard deviation by characteristic strength. */
export function assumedSigma(fck: number): number {
  if (fck <= 15) return 3.5;
  if (fck <= 25) return 4.0;
  return 5.0;
}

/** IS 10262:2019 cl. 3.2 — target mean strength for mix design. */
export function targetMeanStrength(fck: number): number {
  return fck + 1.65 * assumedSigma(fck);
}

/** IS 456:2000 Table 5 — max free water-cement ratio (RCC, 20 mm aggregate). */
const MAX_WC: Record<Exposure, number> = {
  mild: 0.55,
  moderate: 0.5,
  severe: 0.45,
  very_severe: 0.45,
  extreme: 0.4,
};

/** IS 456:2000 Table 5 — minimum cement content (kg/m³, 20 mm aggregate). */
const MIN_CEMENT: Record<Exposure, number> = {
  mild: 300,
  moderate: 300,
  severe: 320,
  very_severe: 340,
  extreme: 360,
};

/** IS 456:2000 Table 5 — minimum grade of concrete for RCC. */
const MIN_GRADE: Record<Exposure, number> = {
  mild: 20,
  moderate: 25,
  severe: 30,
  very_severe: 35,
  extreme: 40,
};

export function maxWcByExposure(e: Exposure): number {
  return MAX_WC[e];
}
export function minCementByExposure(e: Exposure): number {
  return MIN_CEMENT[e];
}
export function minGradeByExposure(e: Exposure): number {
  return MIN_GRADE[e];
}

/** IS 383:2016 Table 4 — fineness modulus zones for fine aggregate. */
export function fmZone(fm: number): { zone: string; ok: boolean } {
  if (fm >= 2.71 && fm <= 3.41) return { zone: 'Zone I', ok: true };
  if (fm >= 2.41 && fm <= 3.1) return { zone: 'Zone II', ok: true };
  if (fm >= 2.11 && fm <= 2.8) return { zone: 'Zone III', ok: true };
  if (fm >= 1.8 && fm <= 2.4) return { zone: 'Zone IV', ok: true };
  return { zone: '—', ok: false };
}

/** IS 456:2000 cl. 6.2.3.1 — short-term static modulus (returned in GPa). */
export function ecFromFck(fck: number): number {
  return 5 * Math.sqrt(Math.max(0, fck));
}

/** IS 456:2000 cl. 6.2.2 — flexural tensile strength (MPa). */
export function flexuralFromFck(fck: number): number {
  return 0.7 * Math.sqrt(Math.max(0, fck));
}

/**
 * Approximate split tensile from fck (MPa). IS 456 does not give an explicit
 * formula; a value of ~0.7·√fck is widely used as a first-cut estimate.
 */
export function splitTensileFromFck(fck: number): number {
  return 0.7 * Math.sqrt(Math.max(0, fck));
}

export interface ComplianceCheck {
  clause: string;
  ok: boolean;
  message: string;
  detail?: string;
}

export interface MixSummary {
  cement: number;
  water: number;
  fineAgg: number;
  coarseAgg: number;
  fm: number;
  copperSlagPct: number;
}

/** Run all available checks for a candidate mix under a chosen exposure class. */
export function complianceReport(mix: MixSummary, exposure: Exposure): ComplianceCheck[] {
  const wc = mix.water / Math.max(1, mix.cement);
  const out: ComplianceCheck[] = [];

  out.push({
    clause: 'IS 456 Table 5 — max w/c',
    ok: wc <= MAX_WC[exposure] + 1e-6,
    message: `w/c = ${wc.toFixed(3)} (limit ${MAX_WC[exposure].toFixed(2)} for ${EXPOSURE_LABELS[exposure]})`,
  });

  out.push({
    clause: 'IS 456 Table 5 — min cement',
    ok: mix.cement >= MIN_CEMENT[exposure] - 1e-6,
    message: `Cement = ${mix.cement.toFixed(0)} kg/m³ (limit ${MIN_CEMENT[exposure]} for ${EXPOSURE_LABELS[exposure]})`,
  });

  const fm = fmZone(mix.fm);
  out.push({
    clause: 'IS 383 — fine agg zone',
    ok: fm.ok,
    message: fm.ok
      ? `FM ${mix.fm.toFixed(2)} → ${fm.zone}`
      : `FM ${mix.fm.toFixed(2)} outside Zones I–IV`,
    detail: fm.zone,
  });

  out.push({
    clause: 'Copper slag replacement',
    ok: mix.copperSlagPct >= 0 && mix.copperSlagPct <= 50,
    message:
      mix.copperSlagPct <= 50
        ? `Copper slag ${mix.copperSlagPct.toFixed(0)}% (≤ 50% of fine agg)`
        : `Copper slag ${mix.copperSlagPct.toFixed(0)}% exceeds the 50% replacement guideline`,
    detail: 'Practical upper bound from published copper-slag concrete studies',
  });

  return out;
}

/** True if ALL checks pass — useful as an optimizer constraint. */
export function isCompliant(mix: MixSummary, exposure: Exposure): boolean {
  return complianceReport(mix, exposure).every((c) => c.ok);
}
