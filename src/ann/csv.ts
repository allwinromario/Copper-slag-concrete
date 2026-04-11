import type { FeatureRow } from './types';

const HEADER_ALIASES: Record<string, keyof ParsedRow> = {
  cement: 'cement',
  water: 'water',
  'copper slag %': 'slag',
  'copper slag': 'slag',
  slag: 'slag',
  'curing days': 'days',
  days: 'days',
  age: 'days',
  'porosity %': 'porosity',
  porosity: 'porosity',
  'crack density': 'crack',
  crack: 'crack',
  'compressive strength': 'strength',
  strength: 'strength',
};

interface ParsedRow {
  cement: number;
  water: number;
  slag: number;
  days: number;
  porosity: number;
  crack: number;
  strength: number;
}

function normalizeHeader(h: string): string {
  return h
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ');
}

/** Parse CSV with header row; flexible column names */
export function parseDataset(csvText: string): {
  X: FeatureRow[];
  y: number[];
  error?: string;
} {
  const lines = csvText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return { X: [], y: [], error: 'Need a header row and at least one data row.' };
  }

  const headerCells = splitCsvLine(lines[0]).map(normalizeHeader);
  const colIndex: Partial<Record<keyof ParsedRow, number>> = {};

  for (let i = 0; i < headerCells.length; i++) {
    const key = HEADER_ALIASES[headerCells[i]];
    if (key) colIndex[key] = i;
  }

  const required: (keyof ParsedRow)[] = [
    'cement',
    'water',
    'slag',
    'days',
    'porosity',
    'crack',
    'strength',
  ];
  const missing = required.filter((k) => colIndex[k] === undefined);
  if (missing.length) {
    return {
      X: [],
      y: [],
      error: `Missing columns: ${missing.join(', ')}. Header was: ${headerCells.join(' | ')}`,
    };
  }

  const X: FeatureRow[] = [];
  const y: number[] = [];

  for (let r = 1; r < lines.length; r++) {
    const cells = splitCsvLine(lines[r]);
    const get = (k: keyof ParsedRow) => {
      const idx = colIndex[k]!;
      const v = parseFloat(cells[idx]?.replace(/,/g, '') ?? '');
      return Number.isFinite(v) ? v : NaN;
    };
    const row: ParsedRow = {
      cement: get('cement'),
      water: get('water'),
      slag: get('slag'),
      days: get('days'),
      porosity: get('porosity'),
      crack: get('crack'),
      strength: get('strength'),
    };
    if (Object.values(row).some((v) => !Number.isFinite(v))) continue;
    X.push([row.cement, row.water, row.slag, row.days, row.porosity, row.crack]);
    y.push(row.strength);
  }

  if (X.length < 4) {
    return {
      X: [],
      y: [],
      error: 'Need at least 4 valid numeric rows after parsing.',
    };
  }

  return { X, y };
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
