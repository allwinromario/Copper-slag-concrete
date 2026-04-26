/**
 * Curated public-CSV sources users can load with one click.
 *
 * Each entry is a stable raw-GitHub mirror so it is fetchable from the
 * browser without CORS issues. Add new entries only after verifying:
 *   1. The URL responds with `Content-Type: text/plain` (raw.githubusercontent.com is fine).
 *   2. The first line is a CSV header recognized by `parseDataset` — at minimum
 *      cement, water, fine aggregate, coarse aggregate, curing days, and
 *      compressive strength.
 */

export interface ExternalDataset {
  id: string;
  label: string;
  url: string;
  caveat: string;
}

export const EXTERNAL_DATASETS: ExternalDataset[] = [
  {
    id: 'uci-concrete',
    label: 'UCI Concrete (Yeh, 2007)',
    url: 'https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Concrete%20Compressive%20Strength.csv',
    caveat:
      '1030 rows. Columns provided: cement, water, fine/coarse aggregate, age, compressive strength. Copper slag %, specific gravities and fineness modulus are NOT in this dataset and will be filled with defaults — only "fck" is meaningfully trained.',
  },
];
