import {
  FEATURE_DEFAULTS,
  N_RAW_FEATURES,
  OUTPUT_KEYS,
  type FeatureRow,
  type OutputKey,
} from './types';

/**
 * Header normalization → input feature index.
 * Keys are pre-normalized (lowercase, units stripped, single spaces).
 */
const FEATURE_ALIASES: Record<string, number> = {
  cement: 0,
  'cement content': 0,
  opc: 0,
  water: 1,
  'water content': 1,
  'fine aggregate': 2,
  'fine agg': 2,
  fine: 2,
  sand: 2,
  'fine aggregate content': 2,
  'coarse aggregate': 3,
  'coarse agg': 3,
  coarse: 3,
  'coarse aggregate content': 3,
  'copper slag %': 4,
  'copper slag': 4,
  copperslag: 4,
  'copper slag percent': 4,
  'curing days': 5,
  days: 5,
  age: 5,
  'curing age': 5,
  'sg cement': 6,
  'specific gravity cement': 6,
  'sg-c': 6,
  'sg fine': 7,
  'sg fine agg': 7,
  'specific gravity fine': 7,
  'sg-f': 7,
  'sg coarse': 8,
  'sg coarse agg': 8,
  'specific gravity coarse': 8,
  'sg-ca': 8,
  fm: 9,
  'fineness modulus': 9,
};

const OUTPUT_ALIASES: Record<string, OutputKey> = {
  'compressive strength': 'fck',
  'concrete compressive strength': 'fck',
  fck: 'fck',
  fc: 'fck',
  "f'c": 'fck',
  strength: 'fck',
  'split tensile strength': 'fst',
  'split tensile': 'fst',
  fst: 'fst',
  'tensile strength': 'fst',
  'flexural strength': 'ffl',
  flexural: 'ffl',
  ffl: 'ffl',
  'modulus of elasticity': 'ec',
  'elastic modulus': 'ec',
  ec: 'ec',
  density: 'density',
  'unit weight': 'density',
};

/** Lowercase, strip units in parens, collapse whitespace. */
function normalizeHeader(h: string): string {
  return h
    .trim()
    .toLowerCase()
    .replace(/\([^)]*\)/g, ' ')
    .replace(/\bkg\/m[³3]\b/g, ' ')
    .replace(/\bmpa\b/g, ' ')
    .replace(/\bgpa\b/g, ' ')
    .replace(/[%]/g, ' %')
    .replace(/\s+/g, ' ')
    .trim();
}

export interface ParsedDataset {
  /** Raw feature rows, length 10 each. Missing columns are filled with defaults. */
  X: FeatureRow[];
  /** Per-output value arrays. Same length as X for keys present in the CSV. */
  y: Partial<Record<OutputKey, number[]>>;
  /** Outputs with at least 4 finite values across rows. */
  availableOutputs: OutputKey[];
  /** Feature columns that were missing in the CSV header (filled with defaults). */
  defaultedFeatures: number[];
  rowCount: number;
  error?: string;
}

const REQUIRED_FEATURE_INDEXES = [0, 1, 2, 3, 5]; // cement, water, fine, coarse, days
const DEFAULTABLE_FEATURE_INDEXES = [4, 6, 7, 8, 9]; // slag, SGs, FM

export function parseDataset(csvText: string): ParsedDataset {
  const empty: ParsedDataset = {
    X: [],
    y: {},
    availableOutputs: [],
    defaultedFeatures: [],
    rowCount: 0,
  };

  const lines = csvText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    return { ...empty, error: 'Need a header row and at least one data row.' };
  }

  const headerCells = splitCsvLine(lines[0]).map(normalizeHeader);
  const featureCol: number[] = Array(N_RAW_FEATURES).fill(-1);
  const outputCol: Partial<Record<OutputKey, number>> = {};

  for (let i = 0; i < headerCells.length; i++) {
    const h = headerCells[i];
    if (h in FEATURE_ALIASES) {
      const idx = FEATURE_ALIASES[h];
      if (featureCol[idx] === -1) featureCol[idx] = i;
    } else if (h in OUTPUT_ALIASES) {
      const key = OUTPUT_ALIASES[h];
      if (outputCol[key] === undefined) outputCol[key] = i;
    }
  }

  const missingRequired = REQUIRED_FEATURE_INDEXES.filter((i) => featureCol[i] === -1);
  if (missingRequired.length) {
    return {
      ...empty,
      error: `Missing required column(s): ${missingRequired
        .map((i) => requiredLabel(i))
        .join(', ')}. Header was: ${headerCells.join(' | ')}`,
    };
  }

  if (outputCol.fck === undefined) {
    return {
      ...empty,
      error: `Missing required output column "Compressive strength". Header was: ${headerCells.join(' | ')}`,
    };
  }

  const defaultedFeatures = DEFAULTABLE_FEATURE_INDEXES.filter((i) => featureCol[i] === -1);

  const X: FeatureRow[] = [];
  const y: Partial<Record<OutputKey, number[]>> = {};
  for (const k of Object.keys(outputCol) as OutputKey[]) y[k] = [];

  for (let r = 1; r < lines.length; r++) {
    const cells = splitCsvLine(lines[r]);

    const row = [...FEATURE_DEFAULTS] as FeatureRow;
    let bad = false;
    for (let i = 0; i < N_RAW_FEATURES; i++) {
      const col = featureCol[i];
      if (col === -1) continue;
      const v = parseFloat(cells[col]?.replace(/,/g, '') ?? '');
      if (!Number.isFinite(v)) {
        if (REQUIRED_FEATURE_INDEXES.includes(i)) {
          bad = true;
          break;
        }
        continue;
      }
      row[i] = v;
    }
    if (bad) continue;

    const fckVal = parseFloat(cells[outputCol.fck!]?.replace(/,/g, '') ?? '');
    if (!Number.isFinite(fckVal)) continue;

    X.push(row);
    for (const k of Object.keys(outputCol) as OutputKey[]) {
      const idx = outputCol[k]!;
      const v = parseFloat(cells[idx]?.replace(/,/g, '') ?? '');
      y[k]!.push(Number.isFinite(v) ? v : NaN);
    }
  }

  if (X.length < 4) {
    return { ...empty, error: 'Need at least 4 valid numeric rows after parsing.' };
  }

  const availableOutputs = OUTPUT_KEYS.filter((k) => {
    const arr = y[k];
    if (!arr) return false;
    return arr.filter((v) => Number.isFinite(v)).length >= 4;
  });

  return {
    X,
    y,
    availableOutputs,
    defaultedFeatures,
    rowCount: X.length,
  };
}

function requiredLabel(i: number): string {
  return ['Cement', 'Water', 'Fine aggregate', 'Coarse aggregate', '', 'Curing days'][i] ?? '';
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (c === ',' && !inQuotes) {
      out.push(cur.trim());
      cur = '';
    } else cur += c;
  }
  out.push(cur.trim());
  return out;
}
