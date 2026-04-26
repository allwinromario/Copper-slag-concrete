import { describe, expect, it } from 'vitest';
import {
  assumedSigma,
  complianceReport,
  ecFromFck,
  EXPOSURE_KEYS,
  flexuralFromFck,
  fmZone,
  isCompliant,
  maxWcByExposure,
  minCementByExposure,
  minGradeByExposure,
  splitTensileFromFck,
  targetMeanStrength,
  type MixSummary,
} from './isCode';

const M30: MixSummary = {
  cement: 360,
  water: 162,
  fineAgg: 720,
  coarseAgg: 1180,
  fm: 2.7,
  copperSlagPct: 20,
};

describe('assumedSigma — IS 10262:2019 Table 1', () => {
  it('returns 3.5 for fck up to 15', () => {
    expect(assumedSigma(10)).toBe(3.5);
    expect(assumedSigma(15)).toBe(3.5);
  });
  it('returns 4.0 for fck 16–25', () => {
    expect(assumedSigma(20)).toBe(4.0);
    expect(assumedSigma(25)).toBe(4.0);
  });
  it('returns 5.0 for fck >= 30', () => {
    expect(assumedSigma(30)).toBe(5.0);
    expect(assumedSigma(60)).toBe(5.0);
  });
});

describe('targetMeanStrength — IS 10262:2019 cl. 3.2', () => {
  it('uses fck + 1.65·σ with the σ from Table 1', () => {
    expect(targetMeanStrength(20)).toBeCloseTo(20 + 1.65 * 4.0, 5);
    expect(targetMeanStrength(30)).toBeCloseTo(30 + 1.65 * 5.0, 5);
    expect(targetMeanStrength(15)).toBeCloseTo(15 + 1.65 * 3.5, 5);
  });
});

describe('IS 456:2000 Table 5 — exposure limits', () => {
  it('maxWcByExposure tightens monotonically up to extreme', () => {
    expect(maxWcByExposure('mild')).toBe(0.55);
    expect(maxWcByExposure('moderate')).toBe(0.5);
    expect(maxWcByExposure('severe')).toBe(0.45);
    expect(maxWcByExposure('very_severe')).toBe(0.45);
    expect(maxWcByExposure('extreme')).toBe(0.4);
  });

  it('minCementByExposure rises with exposure severity', () => {
    expect(minCementByExposure('mild')).toBe(300);
    expect(minCementByExposure('moderate')).toBe(300);
    expect(minCementByExposure('severe')).toBe(320);
    expect(minCementByExposure('very_severe')).toBe(340);
    expect(minCementByExposure('extreme')).toBe(360);
  });

  it('minGradeByExposure follows the M-grade ladder', () => {
    expect(minGradeByExposure('mild')).toBe(20);
    expect(minGradeByExposure('moderate')).toBe(25);
    expect(minGradeByExposure('severe')).toBe(30);
    expect(minGradeByExposure('very_severe')).toBe(35);
    expect(minGradeByExposure('extreme')).toBe(40);
  });

  it('all five exposure keys are exported', () => {
    expect(EXPOSURE_KEYS).toEqual([
      'mild',
      'moderate',
      'severe',
      'very_severe',
      'extreme',
    ]);
  });
});

describe('fmZone — IS 383:2016 Table 4', () => {
  it('classifies typical zones', () => {
    expect(fmZone(2.85).ok).toBe(true);
    expect(fmZone(2.85).zone).toBe('Zone I');
    expect(fmZone(2.6).ok).toBe(true);
    expect(fmZone(2.6).zone).toBe('Zone II');
    expect(fmZone(2.2).ok).toBe(true);
    expect(fmZone(2.2).zone).toBe('Zone III');
    expect(fmZone(1.9).ok).toBe(true);
    expect(fmZone(1.9).zone).toBe('Zone IV');
  });
  it('flags FM outside Zones I–IV', () => {
    expect(fmZone(1.5).ok).toBe(false);
    expect(fmZone(3.6).ok).toBe(false);
  });
});

describe('IS 456 derived strength formulas', () => {
  it('flexural ≈ 0.7·√fck (cl. 6.2.2)', () => {
    expect(flexuralFromFck(25)).toBeCloseTo(0.7 * 5, 5);
    expect(flexuralFromFck(40)).toBeCloseTo(0.7 * Math.sqrt(40), 5);
  });
  it('split tensile uses the same first-cut ≈ 0.7·√fck', () => {
    expect(splitTensileFromFck(40)).toBeCloseTo(flexuralFromFck(40), 5);
  });
  it('Ec ≈ 5·√fck in GPa (cl. 6.2.3.1)', () => {
    expect(ecFromFck(25)).toBeCloseTo(25, 5);
    expect(ecFromFck(40)).toBeCloseTo(5 * Math.sqrt(40), 5);
  });
  it('does not throw on zero or negative fck', () => {
    expect(ecFromFck(0)).toBe(0);
    expect(flexuralFromFck(-5)).toBe(0);
  });
});

describe('complianceReport', () => {
  it('passes a sensible M30 mix at moderate exposure', () => {
    expect(isCompliant(M30, 'moderate')).toBe(true);
  });

  it('flags a too-high w/c for severe exposure', () => {
    const wet = { ...M30, water: 200 };
    const report = complianceReport(wet, 'severe');
    const wc = report.find((c) => c.clause.includes('max w/c'));
    expect(wc).toBeDefined();
    expect(wc!.ok).toBe(false);
    expect(isCompliant(wet, 'severe')).toBe(false);
  });

  it('flags too-low cement content for extreme exposure', () => {
    const lean = { ...M30, cement: 320 };
    const report = complianceReport(lean, 'extreme');
    const cem = report.find((c) => c.clause.includes('min cement'));
    expect(cem).toBeDefined();
    expect(cem!.ok).toBe(false);
  });

  it('flags FM outside any zone', () => {
    const bad = { ...M30, fm: 4.0 };
    const report = complianceReport(bad, 'moderate');
    const fm = report.find((c) => c.clause.includes('fine agg zone'));
    expect(fm).toBeDefined();
    expect(fm!.ok).toBe(false);
  });

  it('flags copper-slag replacement above 50%', () => {
    const overReplaced = { ...M30, copperSlagPct: 60 };
    const report = complianceReport(overReplaced, 'moderate');
    const slag = report.find((c) => c.clause.includes('Copper slag'));
    expect(slag).toBeDefined();
    expect(slag!.ok).toBe(false);
  });

  it('returns one entry per check, in stable order', () => {
    const report = complianceReport(M30, 'moderate');
    expect(report.map((c) => c.clause)).toEqual([
      'IS 456 Table 5 — max w/c',
      'IS 456 Table 5 — min cement',
      'IS 383 — fine agg zone',
      'Copper slag replacement',
    ]);
  });
});
