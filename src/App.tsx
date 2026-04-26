import {
  AnimatePresence,
  LayoutGroup,
  motion,
  useReducedMotion,
  useScroll,
  useTransform,
} from 'framer-motion';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type * as tfTypes from '@tensorflow/tfjs';
import { parseDataset } from './ann/csv';
import { DEMO_CSV, generateDemoData } from './ann/demoData';
import { COPPER_SLAG_TEMPLATE_CSV } from './data/copperSlagTemplate';
import { EXTERNAL_DATASETS, type ExternalDataset } from './data/externalSources';
import {
  buildOptimizedSyntheticCsv,
  EXPOSURE_KEYS,
  OPTIMAL_TRAINING,
  searchMaxStrengthInputs,
  searchMaxStrengthSynthetic,
} from './ann/optimize';
import {
  predictAll,
  trainAnn,
  type TrainedBundle,
} from './ann/trainAnn';
import {
  FEATURE_DEFAULTS,
  FEATURE_LABELS,
  OUTPUT_KEYS,
  OUTPUT_LABELS,
  OUTPUT_UNITS,
  deriveRatios,
  type BatchProgress,
  type EpochLog,
  type FeatureRow,
  type OutputKey,
} from './ann/types';
import { DiagnosticsPanel } from './components/DiagnosticsPanel';
import { MethodologyDialog } from './components/MethodologyDialog';
import { TrainingSidebar } from './components/TrainingSidebar';
import {
  Button,
  Field,
  SectionHeader,
  StatCard,
} from './components/ui';
import {
  loadStoredFeatureInputs,
  persistFeatureInputs,
} from './lib/featureInputsStorage';
import {
  EXPOSURE_LABELS,
  complianceReport,
  type Exposure,
} from './lib/isCode';
import {
  fadeUp,
  springLayout,
  springSoft,
  staggerContainer,
} from './lib/motion';

type DataSource = 'sample' | 'url';

const TABLE_HEADERS = [
  'Cement',
  'Water',
  'Fine',
  'Coarse',
  'Slag %',
  'Days',
  'SGc',
  'SGf',
  'SGca',
  'FM',
  "f'c",
  'fst',
  'fr',
  'Ec',
  'ρ',
] as const;

const INPUT_GROUPS: { title: string; indices: number[] }[] = [
  { title: 'Mix proportions (kg/m³)', indices: [0, 1, 2, 3] },
  { title: 'Mix details', indices: [4, 5] },
  { title: 'Material properties', indices: [6, 7, 8, 9] },
];

const glassPanel =
  'relative overflow-hidden rounded-3xl border border-white/[0.08] bg-white/[0.025] shadow-[0_1px_0_0_rgba(255,255,255,0.04)_inset,0_24px_48px_-24px_rgba(0,0,0,0.6)] backdrop-blur-xl transition-shadow duration-300';

function DatasetPreviewTable({
  parsed,
}: {
  parsed: ReturnType<typeof parseDataset>;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ container: scrollRef });
  const headOpacity = useTransform(scrollYProgress, [0, 0.2], [1, 0.92]);

  if (parsed.error) {
    return (
      <div className="p-4 font-mono text-sm text-red-300/90">{parsed.error}</div>
    );
  }

  if (parsed.X.length === 0) {
    return (
      <div className="p-4 text-sm leading-relaxed text-slate-500">
        No rows parsed yet. Generate sample data, load the demo, or paste CSV in{' '}
        <strong className="text-slate-300">Training setup</strong>.
      </div>
    );
  }

  const total = parsed.X.length;
  const fck = parsed.y.fck ?? [];
  const fst = parsed.y.fst ?? [];
  const ffl = parsed.y.ffl ?? [];
  const ec = parsed.y.ec ?? [];
  const density = parsed.y.density ?? [];

  const cell = (v: number | undefined, dp: number) =>
    v === undefined || !Number.isFinite(v) ? '—' : v.toFixed(dp);

  return (
    <>
      <motion.div
        className="flex shrink-0 items-center justify-between gap-3 border-b border-white/[0.06] bg-white/[0.02] px-3 py-2"
        style={{ opacity: headOpacity }}
      >
        <h3 className="text-[0.6rem] font-medium uppercase tracking-[0.12em] text-slate-400">
          Dataset (CSV)
        </h3>
        <span className="font-mono text-[0.65rem] text-slate-500">
          {total} row{total === 1 ? '' : 's'} · {parsed.availableOutputs.length} output
          {parsed.availableOutputs.length === 1 ? '' : 's'}
        </span>
      </motion.div>
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-auto overscroll-contain [-webkit-overflow-scrolling:touch]"
      >
        <table className="w-full border-collapse font-mono text-[0.7rem]">
          <thead>
            <tr className="sticky top-0 z-[1] border-b border-white/[0.08] bg-[#0a0b10]/95 backdrop-blur-md">
              <th className="w-10 px-2 py-2 text-center text-[0.6rem] font-medium uppercase tracking-[0.1em] text-slate-500">
                #
              </th>
              {TABLE_HEADERS.map((h) => (
                <th
                  key={h}
                  scope="col"
                  title={h}
                  className="whitespace-nowrap px-2 py-2 text-right text-[0.6rem] font-medium uppercase tracking-[0.1em] text-slate-500"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {parsed.X.map((row, i) => (
              <motion.tr
                key={i}
                initial={false}
                className="border-b border-white/[0.04] transition-colors duration-150 odd:bg-white/[0.015] hover:bg-cyan-400/[0.05]"
              >
                <td className="px-2 py-1.5 text-center text-slate-500">{i + 1}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[0].toFixed(0)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[1].toFixed(0)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[2].toFixed(0)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[3].toFixed(0)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[4].toFixed(0)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[5]}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[6].toFixed(2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[7].toFixed(2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[8].toFixed(2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[9].toFixed(2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{cell(fck[i], 2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-300">{cell(fst[i], 2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-300">{cell(ffl[i], 2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-300">{cell(ec[i], 2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-300">{cell(density[i], 0)}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

interface DataColumnProps {
  dataSource: DataSource;
  syntheticRowCount: number;
  setSyntheticRowCount: (n: number) => void;
  loadSynthetic: () => void;
  setCsvText: (s: string) => void;
  setError: (e: string | null) => void;
  parsed: ReturnType<typeof parseDataset>;
  markDatasetPrimed: () => void;
  onOptimizeSetup: () => void;
  loadCopperSlagTemplate: () => void;
  fetchUrl: string;
  setFetchUrl: (s: string) => void;
  fetchingUrl: boolean;
  loadFromUrl: (url?: string) => void;
  loadExternal: (d: ExternalDataset) => void;
}

function DataColumn(p: DataColumnProps) {
  const {
    dataSource,
    syntheticRowCount,
    setSyntheticRowCount,
    loadSynthetic,
    setCsvText,
    setError,
    parsed,
    markDatasetPrimed,
    onOptimizeSetup,
    loadCopperSlagTemplate,
    fetchUrl,
    setFetchUrl,
    fetchingUrl,
    loadFromUrl,
    loadExternal,
  } = p;

  return (
    <>
      <SectionHeader eyebrow="Data source" title="Training data" />

      <div className="shrink-0">
        {dataSource === 'sample' && (
          <>
            <p className="mb-3 text-[0.8rem] leading-relaxed text-slate-400">
              Generate plausible rows from the synthetic recipe, load the small demo, or
              paste a CSV in <strong className="text-slate-200">Training setup</strong>.
            </p>
            <div className="mb-3 flex flex-wrap items-end gap-2.5">
              <div className="w-[6.5rem]">
                <Field
                  label="Rows"
                  type="number"
                  min={20}
                  max={500}
                  value={syntheticRowCount}
                  onChange={(e) => setSyntheticRowCount(Number(e.target.value))}
                />
              </div>
              <Button
                variant="primary"
                onClick={() => {
                  loadSynthetic();
                  markDatasetPrimed();
                }}
              >
                Generate
              </Button>
              <Button
                variant="secondary"
                onClick={() => {
                  setCsvText(DEMO_CSV);
                  setError(null);
                  markDatasetPrimed();
                }}
              >
                Small demo
              </Button>
              <Button
                variant="secondary"
                onClick={loadCopperSlagTemplate}
                title="Load the copper-slag CSV header with a few starter rows you can edit and replace with lab data"
              >
                Copper-slag template
              </Button>
              <Button
                variant="accent"
                glow
                onClick={onOptimizeSetup}
                title="Larger synthetic set + tuned epochs, learning rate, and validation split"
              >
                Optimize setup
              </Button>
            </div>
            <div className="mb-2 flex flex-wrap gap-2">
              {parsed.error ? (
                <span className="inline-flex rounded-full border border-red-400/25 bg-red-500/10 px-3 py-1 font-mono text-[0.7rem] text-red-300">
                  {parsed.error}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/20 bg-emerald-500/[0.08] px-3 py-1 font-mono text-[0.7rem] text-emerald-300">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  {parsed.X.length} row{parsed.X.length === 1 ? '' : 's'} ·{' '}
                  {parsed.availableOutputs.length} output
                  {parsed.availableOutputs.length === 1 ? '' : 's'}
                </span>
              )}
              {parsed.defaultedFeatures.length > 0 && (
                <span className="inline-flex rounded-full border border-amber-400/20 bg-amber-500/[0.08] px-3 py-1 font-mono text-[0.7rem] text-amber-200/90">
                  {parsed.defaultedFeatures.length} column(s) defaulted
                </span>
              )}
            </div>
          </>
        )}

        {dataSource === 'url' && (
          <>
            <p className="mb-3 text-[0.8rem] leading-relaxed text-slate-400">
              Paste a raw CSV URL (GitHub raw, gist, or any CORS-friendly host).
              Required columns:{' '}
              <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[0.7rem] text-slate-300">
                Cement, Water, Fine aggregate, Coarse aggregate, Curing days, Compressive strength
              </code>
              . Other columns are optional — missing inputs are filled with sensible
              defaults.
            </p>
            <div className="mb-3 flex flex-wrap items-end gap-2.5">
              <div className="min-w-[18rem] flex-1">
                <Field
                  label="CSV URL"
                  type="url"
                  placeholder="https://raw.githubusercontent.com/.../dataset.csv"
                  value={fetchUrl}
                  onChange={(e) => setFetchUrl(e.target.value)}
                />
              </div>
              <Button
                variant="primary"
                disabled={fetchingUrl || !fetchUrl}
                loading={fetchingUrl}
                onClick={() => loadFromUrl()}
              >
                {fetchingUrl ? 'Fetching…' : 'Fetch CSV'}
              </Button>
            </div>

            <div className="mb-2">
              <p className="mb-2 text-[0.6rem] font-medium uppercase tracking-[0.12em] text-slate-500">
                One-click public datasets
              </p>
              <div className="flex flex-wrap gap-2">
                {EXTERNAL_DATASETS.map((d) => (
                  <Button
                    key={d.id}
                    variant="accent"
                    disabled={fetchingUrl}
                    onClick={() => loadExternal(d)}
                    title={d.caveat}
                  >
                    {d.label}
                  </Button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

      <div className="mt-3 flex min-h-0 flex-1 flex-col">
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-white/[0.06] bg-black/30">
          <DatasetPreviewTable parsed={parsed} />
        </div>
      </div>
    </>
  );
}

function DatasetDialog({
  open,
  onClose,
  parsed,
}: {
  open: boolean;
  onClose: () => void;
  parsed: ReturnType<typeof parseDataset>;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4 py-6 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          role="dialog"
          aria-modal="true"
          aria-label="Parsed CSV preview"
        >
          <motion.div
            className="relative flex h-[min(85vh,720px)] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-slate-950/97 shadow-2xl shadow-black/60"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 320, damping: 28 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex shrink-0 items-start justify-between gap-4 border-b border-white/[0.06] px-6 py-4">
              <div>
                <p className="text-[0.6rem] font-semibold uppercase tracking-[0.16em] text-cyan-300/90">
                  Training data
                </p>
                <h2 className="mt-1 text-lg font-semibold text-white">
                  Parsed CSV preview
                </h2>
                <p className="mt-1 text-[0.7rem] leading-relaxed text-slate-500">
                  {parsed.X.length} row{parsed.X.length === 1 ? '' : 's'} ·{' '}
                  {parsed.availableOutputs.length} output column
                  {parsed.availableOutputs.length === 1 ? '' : 's'} present
                  {parsed.availableOutputs.length > 0
                    ? ` (${parsed.availableOutputs.join(', ')})`
                    : ''}
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close"
                className="rounded-md border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-white/30 hover:text-white"
              >
                Close
              </button>
            </div>
            <div className="flex min-h-0 flex-1 flex-col">
              <DatasetPreviewTable parsed={parsed} />
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function RatioField({
  label,
  hint,
  value,
  onCommit,
}: {
  label: string;
  hint: string;
  value: number;
  onCommit: (next: number) => void;
}) {
  const [leftText, setLeftText] = useState('1');
  const [rightText, setRightText] = useState(() => formatRatioDenom(value));
  const focusCountRef = useRef(0);

  useEffect(() => {
    if (focusCountRef.current > 0) return;
    const num = Number.parseFloat(leftText);
    const den = Number.parseFloat(rightText);
    if (
      Number.isFinite(num) &&
      Number.isFinite(den) &&
      num > 0 &&
      den > 0 &&
      Math.abs(num / den - value) <= 5e-4
    ) {
      return;
    }
    setLeftText('1');
    setRightText(formatRatioDenom(value));
  }, [value]);

  const tryCommit = (l: string, r: string) => {
    const num = Number.parseFloat(l);
    const den = Number.parseFloat(r);
    if (Number.isFinite(num) && Number.isFinite(den) && num > 0 && den > 0) {
      onCommit(num / den);
    }
  };

  const handleFocus = () => {
    focusCountRef.current += 1;
  };

  const handleBlur = () => {
    focusCountRef.current = Math.max(0, focusCountRef.current - 1);
    if (focusCountRef.current > 0) return;
    const num = Number.parseFloat(leftText);
    const den = Number.parseFloat(rightText);
    if (!Number.isFinite(num) || !Number.isFinite(den) || num <= 0 || den <= 0) {
      setLeftText('1');
      setRightText(formatRatioDenom(value));
    }
  };

  const inputClass = [
    'w-full min-w-0 rounded-xl border border-white/10 bg-black/40 px-3 py-2.5 font-mono text-sm text-white text-center',
    'placeholder:text-slate-600',
    'transition-[border-color,box-shadow,background-color] duration-200',
    'hover:border-white/20',
    'focus:border-cyan-400/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/25',
  ].join(' ');

  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <label className="text-[0.65rem] font-medium uppercase tracking-[0.1em] text-slate-500">
        {label}
      </label>
      <div className="flex min-w-0 items-center gap-2">
        <input
          type="text"
          inputMode="decimal"
          autoComplete="off"
          spellCheck={false}
          aria-label={`${label} — first number`}
          className={inputClass}
          value={leftText}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onChange={(e) => {
            const v = e.target.value;
            setLeftText(v);
            tryCommit(v, rightText);
          }}
        />
        <span
          aria-hidden
          className="select-none font-mono text-base font-semibold text-slate-400"
        >
          :
        </span>
        <input
          type="text"
          inputMode="decimal"
          autoComplete="off"
          spellCheck={false}
          aria-label={`${label} — second number`}
          className={inputClass}
          value={rightText}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onChange={(e) => {
            const v = e.target.value;
            setRightText(v);
            tryCommit(leftText, v);
          }}
        />
      </div>
      {hint && <p className="text-[0.65rem] leading-relaxed text-slate-500">{hint}</p>}
    </div>
  );
}

/**
 * Given a positive ratio = a/b, return the canonical denominator value to
 * display when the numerator is shown as 1 (i.e. b/a, rounded to 3 decimals).
 */
function formatRatioDenom(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0';
  return (1 / value).toFixed(3);
}

export default function App() {
  const reduce = useReducedMotion();
  const [mqWide, setMqWide] = useState(true);

  const [dataSource, setDataSource] = useState<DataSource>('sample');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [datasetDialogOpen, setDatasetDialogOpen] = useState(false);
  const [csvText, setCsvText] = useState(DEMO_CSV);
  const [syntheticRowCount, setSyntheticRowCount] = useState(120);
  const [epochs, setEpochs] = useState(260);
  const [lr, setLr] = useState(0.01);
  const [valFrac, setValFrac] = useState(0.2);
  const [epochLogs, setEpochLogs] = useState<EpochLog[]>([]);
  const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<TrainedBundle | null>(null);
  const [inputs, setInputs] = useState<FeatureRow>(() =>
    loadStoredFeatureInputs([...FEATURE_DEFAULTS] as FeatureRow)
  );
  const [prediction, setPrediction] = useState<Partial<Record<OutputKey, number>> | null>(null);
  const [exposure, setExposure] = useState<Exposure>('moderate');
  const [fetchUrl, setFetchUrl] = useState('');
  const [fetchingUrl, setFetchingUrl] = useState(false);
  const [methodologyOpen, setMethodologyOpen] = useState(false);

  const [datasetPrimed, setDatasetPrimed] = useState(false);
  const [hasSavedInputs, setHasSavedInputs] = useState(false);
  const [threeColLayout, setThreeColLayout] = useState(false);
  const [optimizingInputs, setOptimizingInputs] = useState(false);

  const modelRef = useRef<tfTypes.Sequential | null>(null);
  const optimizeInputsBusy = useRef(false);
  const mainScrollRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ container: mainScrollRef });
  const heroY = useTransform(scrollYProgress, [0, 0.15], [0, -8]);

  useEffect(() => {
    const mq = window.matchMedia('(min-width: 900px)');
    const apply = () => setMqWide(mq.matches);
    apply();
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, []);

  const parsed = useMemo(() => parseDataset(csvText), [csvText]);

  const compliance = useMemo(
    () =>
      complianceReport(
        {
          cement: inputs[0],
          water: inputs[1],
          fineAgg: inputs[2],
          coarseAgg: inputs[3],
          fm: inputs[9],
          copperSlagPct: inputs[4],
        },
        exposure
      ),
    [inputs, exposure]
  );

  const markDatasetPrimed = useCallback(() => {
    setDatasetPrimed(true);
    setHasSavedInputs(false);
  }, []);

  const loadSynthetic = useCallback(() => {
    const n = Math.min(500, Math.max(20, Math.round(syntheticRowCount)));
    const { X, y } = generateDemoData(n);
    const header =
      'Cement,Water,Fine aggregate,Coarse aggregate,Copper slag %,Curing days,SG cement,SG fine,SG coarse,Fineness modulus,Compressive strength,Split tensile,Flexural strength,Modulus of elasticity,Density';
    const rows = X.map(
      (r, i) =>
        [
          r[0].toFixed(1),
          r[1].toFixed(1),
          r[2].toFixed(1),
          r[3].toFixed(1),
          r[4].toFixed(1),
          r[5],
          r[6].toFixed(2),
          r[7].toFixed(2),
          r[8].toFixed(2),
          r[9].toFixed(2),
          y.fck[i].toFixed(2),
          y.fst[i].toFixed(2),
          y.ffl[i].toFixed(2),
          y.ec[i].toFixed(2),
          y.density[i].toFixed(0),
        ].join(',')
    );
    setCsvText([header, ...rows].join('\n'));
    setError(null);
  }, [syntheticRowCount]);

  const loadFromUrl = useCallback(
    async (overrideUrl?: string) => {
      const target = overrideUrl ?? fetchUrl;
      if (!target) return;
      setFetchingUrl(true);
      setError(null);
      try {
        const res = await fetch(target);
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        const text = await res.text();
        setCsvText(text);
        markDatasetPrimed();
      } catch (e) {
        setError(
          e instanceof Error
            ? `Could not fetch CSV: ${e.message}. The host may not allow cross-origin requests; try a GitHub raw URL.`
            : String(e)
        );
      } finally {
        setFetchingUrl(false);
      }
    },
    [fetchUrl, markDatasetPrimed]
  );

  const loadCopperSlagTemplate = useCallback(() => {
    setCsvText(COPPER_SLAG_TEMPLATE_CSV);
    setError(null);
    markDatasetPrimed();
  }, [markDatasetPrimed]);

  const loadExternal = useCallback(
    (d: ExternalDataset) => {
      setFetchUrl(d.url);
      void loadFromUrl(d.url);
    },
    [loadFromUrl]
  );

  const showTrainColumn = datasetPrimed && hasSavedInputs && !threeColLayout;
  const workspacePhase = threeColLayout ? 'p3' : showTrainColumn ? 'p2' : 'p1';

  const gridCols = useMemo(() => {
    if (!mqWide) return '1fr';
    if (workspacePhase === 'p1') return 'minmax(0,1fr) 0px minmax(0,1fr)';
    if (workspacePhase === 'p2')
      return 'minmax(0,1fr) minmax(160px,min(220px,14vw)) minmax(0,1fr)';
    return 'minmax(0,1fr) minmax(0,1fr) minmax(0,1fr)';
  }, [mqWide, workspacePhase]);

  useEffect(() => {
    if (!sidebarOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSidebarOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [sidebarOpen]);

  useEffect(() => {
    persistFeatureInputs(inputs);
  }, [inputs]);

  const train = async () => {
    if (parsed.error || parsed.X.length === 0) {
      setError(parsed.error ?? 'No data');
      return;
    }
    setError(null);
    setTraining(true);
    setEpochLogs([]);
    setBatchProgress(null);
    setBundle(null);
    setPrediction(null);

    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }

    try {
      const result = await trainAnn(
        {
          X: parsed.X,
          y: parsed.y,
          availableOutputs: parsed.availableOutputs,
        },
        {
          epochs,
          learningRate: lr,
          validationFraction: valFrac,
          onEpoch: (log) => {
            setEpochLogs((prev) => [...prev, log]);
          },
          onBatch: (p) => {
            setBatchProgress(p);
          },
        }
      );
      modelRef.current = result.model;
      setBundle(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setTraining(false);
      setBatchProgress(null);
    }
  };

  const handleMainTrain = async () => {
    setThreeColLayout(true);
    await train();
  };

  const runPredict = () => {
    if (!bundle) return;
    const v = predictAll(bundle.model, bundle.norm, inputs);
    setPrediction(v);
  };

  const saveInputs = () => {
    setHasSavedInputs(true);
  };

  const updateInput = (i: number, value: number) => {
    setHasSavedInputs(false);
    setInputs((prev) => {
      const next = [...prev] as FeatureRow;
      next[i] = value;
      return next;
    });
  };

  const ratios = useMemo(() => deriveRatios(inputs), [inputs]);

  const updateRatio = useCallback((which: 0 | 1 | 2 | 3, value: number) => {
    if (!Number.isFinite(value) || value <= 0) return;
    setHasSavedInputs(false);
    setInputs((prev) => {
      const next = [...prev] as FeatureRow;
      const cement = next[0];
      const fine = next[2];
      switch (which) {
        case 0:
          next[1] = value * cement;
          break;
        case 1:
          next[2] = cement / value;
          break;
        case 2:
          next[3] = cement / value;
          break;
        case 3:
          next[3] = fine / value;
          break;
      }
      return next;
    });
  }, []);

  const applyOptimalTrainingSetup = useCallback(() => {
    setSyntheticRowCount(OPTIMAL_TRAINING.syntheticRowCount);
    setEpochs(OPTIMAL_TRAINING.epochs);
    setLr(OPTIMAL_TRAINING.learningRate);
    setValFrac(OPTIMAL_TRAINING.validationFraction);
    setCsvText(buildOptimizedSyntheticCsv());
    setError(null);
    markDatasetPrimed();
  }, [markDatasetPrimed]);

  const applyOptimalHyperparams = useCallback(() => {
    setEpochs(OPTIMAL_TRAINING.epochs);
    setLr(OPTIMAL_TRAINING.learningRate);
    setValFrac(OPTIMAL_TRAINING.validationFraction);
  }, []);

  const applyOptimizeInferenceInputs = useCallback(async () => {
    if (optimizeInputsBusy.current) return;
    optimizeInputsBusy.current = true;
    setOptimizingInputs(true);
    try {
      const next = bundle
        ? await searchMaxStrengthInputs(bundle.model, bundle.norm, exposure)
        : await searchMaxStrengthSynthetic(exposure);
      setInputs(next);
      setHasSavedInputs(false);
    } finally {
      optimizeInputsBusy.current = false;
      setOptimizingInputs(false);
    }
  }, [bundle, exposure]);

  const refreshApp = useCallback(() => {
    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }
    setSidebarOpen(false);
    setDataSource('sample');
    setCsvText(DEMO_CSV);
    setSyntheticRowCount(120);
    setEpochs(260);
    setLr(0.01);
    setValFrac(0.2);
    setEpochLogs([]);
    setTraining(false);
    setError(null);
    setBundle(null);
    setPrediction(null);
    setDatasetPrimed(false);
    setHasSavedInputs(false);
    setThreeColLayout(false);
    optimizeInputsBusy.current = false;
    setOptimizingInputs(false);
  }, []);

  const gridTransition = reduce ? { duration: 0.15 } : springLayout;

  return (
    <div
      className="relative isolate flex h-full max-h-[100dvh] min-h-0 flex-col overflow-hidden bg-[#07080c] text-slate-100"
      data-phase={workspacePhase}
    >
      <MethodologyDialog
        open={methodologyOpen}
        onClose={() => setMethodologyOpen(false)}
      />
      <div
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_85%_55%_at_0%_-10%,rgba(212,184,150,0.14),transparent_52%),radial-gradient(ellipse_70%_45%_at_100%_0%,rgba(45,212,191,0.08),transparent_48%),radial-gradient(ellipse_60%_40%_at_50%_110%,rgba(15,118,110,0.12),transparent_55%)]"
        aria-hidden
      />
      <div
        className="pointer-events-none absolute inset-0 -z-10 opacity-[0.035]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
        }}
        aria-hidden
      />

      <LayoutGroup>
        <div
          ref={mainScrollRef}
          className="relative z-[1] flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-hidden px-[clamp(1rem,4vw,3.75rem)] py-[clamp(0.5rem,1.8vh,1.35rem)]"
        >
          <motion.div
            className="mb-5 flex shrink-0 flex-wrap items-end justify-between gap-4 border-b border-white/[0.06] pb-4"
            style={{ y: heroY }}
          >
            <header className="min-w-0 flex-1">
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
                className="mb-1.5 inline-flex items-center gap-2 text-[0.6rem] font-medium uppercase tracking-[0.16em] text-slate-500"
              >
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-cyan-400/80 shadow-[0_0_10px_rgba(62,232,214,0.55)]" />
                ANN mix-design model
              </motion.div>
              <motion.h1
                className="text-[clamp(1.5rem,1.35rem+1.6vw,2.5rem)] font-semibold leading-[1.1] tracking-tight text-white"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={reduce ? { duration: 0.2 } : { duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              >
                Copper slag concrete
              </motion.h1>
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4, delay: 0.1 }}
                className="mt-2 max-w-[min(46rem,60vw)] text-[clamp(0.8125rem,0.75rem+0.3vw,0.9rem)] leading-relaxed text-slate-400"
              >
                In-browser TensorFlow.js training with IS 456 / 10262 / 383 reference
                checks. Predicts compressive, split tensile, flexural, modulus of
                elasticity, and density.
              </motion.p>
            </header>
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.12, ease: [0.4, 0, 0.2, 1] }}
              className="flex shrink-0 flex-wrap items-end gap-2"
            >
              <label className="flex flex-col gap-1.5 text-[0.6rem] font-medium uppercase tracking-[0.12em] text-slate-500">
                Exposure
                <select
                  value={exposure}
                  onChange={(e) => setExposure(e.target.value as Exposure)}
                  className="h-10 rounded-xl border border-white/10 bg-black/40 px-3 font-mono text-xs text-slate-100 transition-colors hover:border-white/20 focus:border-cyan-400/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/25"
                >
                  {EXPOSURE_KEYS.map((k) => (
                    <option key={k} value={k}>
                      {EXPOSURE_LABELS[k]}
                    </option>
                  ))}
                </select>
              </label>
              <Button
                variant="secondary"
                onClick={refreshApp}
                aria-label="Reset model and training data"
                title="Reset model and data (inference numbers stay saved in this browser)"
              >
                <svg
                  className="h-3.5 w-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden
                >
                  <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                  <path d="M3 3v5h5" />
                  <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                  <path d="M16 16h5v5" />
                </svg>
                Refresh
              </Button>
              <Button
                variant="primary"
                onClick={() => setSidebarOpen(true)}
                aria-expanded={sidebarOpen}
              >
                <span className="h-1.5 w-1.5 rounded-full bg-slate-950" aria-hidden />
                Training setup
              </Button>
            </motion.div>
          </motion.div>

          <motion.section
            className={`${glassPanel} mb-4 shrink-0 px-[clamp(1rem,2.5vw,1.5rem)] py-[clamp(0.9rem,2vw,1.15rem)]`}
            variants={staggerContainer}
            initial="hidden"
            animate="show"
            aria-label="Data source selection"
          >
            <SectionHeader
              eyebrow="Data source"
              title="Build your training table"
              className="mb-3"
            />
            <motion.div
              variants={fadeUp}
              role="tablist"
              aria-label="Data source"
              className="flex gap-1 rounded-xl border border-white/[0.06] bg-black/30 p-1"
            >
              {(['sample', 'url'] as const).map((tab) => (
                <motion.button
                  key={tab}
                  type="button"
                  role="tab"
                  aria-selected={dataSource === tab}
                  onClick={() => setDataSource(tab)}
                  className={`relative flex-1 rounded-lg px-4 py-2.5 text-sm font-medium tracking-tight transition-colors ${
                    dataSource === tab
                      ? 'text-white'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}
                  whileTap={{ scale: 0.98 }}
                >
                  {dataSource === tab && (
                    <motion.span
                      layoutId="tab-pill"
                      className="absolute inset-0 -z-10 rounded-lg bg-white/[0.06] ring-1 ring-white/10"
                      transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                    />
                  )}
                  {tab === 'sample' ? 'Sample data' : 'Fetch CSV from URL'}
                </motion.button>
              ))}
            </motion.div>
          </motion.section>

          <motion.div
            className="grid min-h-0 min-h-[200px] flex-1 gap-[clamp(0.75rem,1.8vw,1.35rem)] overflow-hidden pb-1"
            animate={{ gridTemplateColumns: gridCols }}
            transition={gridTransition}
          >
            <motion.section
              layout
              className={`${glassPanel} flex min-h-0 min-w-0 flex-col px-[clamp(0.9rem,2vw,1.35rem)] py-[clamp(0.85rem,1.8vw,1.2rem)]`}
              aria-label="Data source and dataset"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ ...springSoft, delay: 0.05 }}
            >
              <DataColumn
                dataSource={dataSource}
                syntheticRowCount={syntheticRowCount}
                setSyntheticRowCount={setSyntheticRowCount}
                loadSynthetic={loadSynthetic}
                setCsvText={setCsvText}
                setError={setError}
                parsed={parsed}
                markDatasetPrimed={markDatasetPrimed}
                onOptimizeSetup={applyOptimalTrainingSetup}
                loadCopperSlagTemplate={loadCopperSlagTemplate}
                fetchUrl={fetchUrl}
                setFetchUrl={setFetchUrl}
                fetchingUrl={fetchingUrl}
                loadFromUrl={loadFromUrl}
                loadExternal={loadExternal}
              />
            </motion.section>

            <motion.section
              layout
              className={`flex min-h-0 min-w-0 flex-col overflow-hidden ${glassPanel} px-[clamp(0.9rem,2vw,1.35rem)] py-[clamp(0.85rem,1.8vw,1.2rem)] ${
                workspacePhase === 'p1'
                  ? 'pointer-events-none invisible m-0 max-w-0 min-w-0 w-0 border-0 p-0 opacity-0 shadow-none ring-0 max-lg:hidden'
                  : ''
              }`}
              aria-hidden={workspacePhase === 'p1'}
            >
              <AnimatePresence mode="wait">
                {workspacePhase === 'p2' && (
                  <motion.div
                    key="train"
                    className="flex min-h-0 flex-1 flex-col items-stretch justify-center"
                    initial={{ opacity: 0, scale: 0.96, filter: 'blur(6px)' }}
                    animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                    exit={{ opacity: 0, scale: 0.97 }}
                    transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <SectionHeader eyebrow="Train" title="Ready to train" />
                    <p className="mb-6 text-[0.85rem] leading-relaxed text-slate-400">
                      Dataset loaded and inputs saved. Tune epochs &amp; learning rate in{' '}
                      <strong className="text-slate-200">Training setup</strong> if needed.
                    </p>
                    <Button
                      variant="primary"
                      size="lg"
                      fullWidth
                      disabled={training}
                      loading={training}
                      onClick={handleMainTrain}
                    >
                      {training ? 'Training…' : 'Train ANN'}
                    </Button>
                  </motion.div>
                )}
                {workspacePhase === 'p3' && (
                  <motion.div
                    key="diag"
                    className="flex min-h-0 flex-1 flex-col gap-3"
                    initial={{ opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                  >
                    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-white/[0.015]">
                      <DiagnosticsPanel
                        training={training}
                        error={error}
                        epochLogs={epochLogs}
                        bundle={bundle}
                        batchProgress={batchProgress}
                      />
                    </div>
                    <Button
                      variant="primary"
                      size="lg"
                      fullWidth
                      glow
                      disabled={!bundle || training}
                      onClick={runPredict}
                    >
                      Generate prediction
                    </Button>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.section>

            <motion.section
              layout
              className={`${glassPanel} flex min-h-0 min-w-0 flex-col px-[clamp(0.9rem,2vw,1.35rem)] py-[clamp(0.85rem,1.8vw,1.2rem)]`}
              aria-label="Inference"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ ...springSoft, delay: 0.1 }}
            >
              <SectionHeader
                eyebrow="Inference"
                title="Predict concrete properties"
              />
              <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain pr-0.5">
                {threeColLayout && (
                  <p className="mb-4 text-[0.8rem] leading-relaxed text-slate-400">
                    Adjust inputs, then click <strong className="text-slate-200">Generate
                    prediction</strong> in Diagnostics — outputs animate in below the
                    workspace.
                  </p>
                )}
                {INPUT_GROUPS.map((group, gi) => (
                  <motion.div
                    key={group.title}
                    className="mb-5"
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.05 + gi * 0.05, duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                  >
                    <p className="mb-2 text-[0.6rem] font-medium uppercase tracking-[0.12em] text-slate-500">
                      {group.title}
                    </p>
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      {group.indices.map((i) => (
                        <Field
                          key={FEATURE_LABELS[i]}
                          label={FEATURE_LABELS[i]}
                          type="number"
                          step="any"
                          value={inputs[i]}
                          onChange={(e) => updateInput(i, Number(e.target.value))}
                        />
                      ))}
                    </div>
                  </motion.div>
                ))}

                <motion.div
                  className="mb-5"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2, duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                >
                  <div className="mb-2 flex items-baseline justify-between gap-2">
                    <p className="text-[0.6rem] font-medium uppercase tracking-[0.12em] text-slate-500">
                      Mix ratios
                    </p>
                    <p className="text-[0.55rem] font-medium uppercase tracking-[0.16em] text-slate-600">
                      enter as <span className="font-mono text-slate-400">a&nbsp;:&nbsp;b</span>
                    </p>
                  </div>
                  <div className="grid grid-cols-1 gap-4">
                    <RatioField
                      label="w/c ratio"
                      hint="Updates Water"
                      value={ratios[0]}
                      onCommit={(v) => updateRatio(0, v)}
                    />
                    <RatioField
                      label="Cement / Fine agg"
                      hint="Updates Fine aggregate"
                      value={ratios[1]}
                      onCommit={(v) => updateRatio(1, v)}
                    />
                    <RatioField
                      label="Cement / Coarse agg"
                      hint="Updates Coarse aggregate"
                      value={ratios[2]}
                      onCommit={(v) => updateRatio(2, v)}
                    />
                    <RatioField
                      label="Fine / Coarse agg"
                      hint="Updates Coarse aggregate"
                      value={ratios[3]}
                      onCommit={(v) => updateRatio(3, v)}
                    />
                  </div>
                </motion.div>

                <motion.div
                  layout
                  className="mb-4 rounded-2xl border border-white/[0.08] bg-black/20 p-3.5"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.25, duration: 0.35 }}
                >
                  <div className="mb-2.5 flex items-center justify-between gap-2">
                    <p className="text-[0.6rem] font-medium uppercase tracking-[0.12em] text-amber-200/85">
                      IS code check · {EXPOSURE_LABELS[exposure]}
                    </p>
                    <button
                      type="button"
                      onClick={() => setMethodologyOpen(true)}
                      className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[0.58rem] font-medium uppercase tracking-[0.12em] text-slate-400 transition-colors hover:border-white/25 hover:text-slate-100"
                    >
                      View references
                    </button>
                  </div>
                  <ul className="flex flex-col gap-1.5 font-mono text-[0.7rem]">
                    {compliance.map((c) => (
                      <motion.li
                        key={c.clause}
                        layout
                        initial={false}
                        animate={{ opacity: 1 }}
                        className={`flex items-start gap-2 ${
                          c.ok ? 'text-emerald-300/90' : 'text-amber-300/95'
                        }`}
                      >
                        <span aria-hidden className="mt-0.5">
                          {c.ok ? '✓' : '!'}
                        </span>
                        <span>
                          <strong className="font-medium text-slate-300">
                            {c.clause}:
                          </strong>{' '}
                          {c.message}
                        </span>
                      </motion.li>
                    ))}
                  </ul>
                </motion.div>

                <div className="mt-auto flex flex-wrap items-center gap-2 pt-1">
                  <Button
                    variant="accent"
                    disabled={training || optimizingInputs}
                    loading={optimizingInputs}
                    onClick={() => void applyOptimizeInferenceInputs()}
                    title={
                      bundle
                        ? `Search for IS-compliant (${EXPOSURE_LABELS[exposure]}) inputs that maximize ANN strength`
                        : `Suggest a high-strength IS-compliant (${EXPOSURE_LABELS[exposure]}) mix using the demo formula`
                    }
                  >
                    {optimizingInputs ? 'Optimizing…' : 'Optimize inputs (IS-compliant)'}
                  </Button>
                  <Button variant="secondary" onClick={saveInputs}>
                    Save inputs
                  </Button>
                  <AnimatePresence>
                    {hasSavedInputs && (
                      <motion.span
                        initial={{ opacity: 0, scale: 0.95, x: -4 }}
                        animate={{ opacity: 1, scale: 1, x: 0 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
                        className="inline-flex items-center gap-1.5 rounded-full border border-emerald-400/25 bg-emerald-500/10 px-3 py-1 font-mono text-[0.7rem] text-emerald-300"
                      >
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                        Inputs saved
                      </motion.span>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </motion.section>
          </motion.div>

          <AnimatePresence>
            {threeColLayout && prediction != null && (
              <motion.section
                layout
                className={`${glassPanel} relative mt-3 shrink-0 px-4 py-4 sm:px-5 sm:py-4`}
                aria-label={
                  prediction.fck != null
                    ? `Predicted compressive strength ${prediction.fck.toFixed(2)} megapascals`
                    : 'Predicted concrete properties'
                }
                aria-live="polite"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 8 }}
                transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
              >
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-[0.6rem] font-medium uppercase tracking-[0.14em] text-slate-500">
                      Prediction
                    </p>
                    <p className="mt-0.5 text-[0.85rem] font-medium text-slate-200">
                      ANN output for the current mix
                    </p>
                  </div>
                  <motion.button
                    type="button"
                    className="flex h-8 w-8 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-slate-400 transition-colors hover:border-white/25 hover:text-slate-100"
                    onClick={() => setPrediction(null)}
                    aria-label="Dismiss prediction"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <span className="text-base leading-none" aria-hidden>
                      ×
                    </span>
                  </motion.button>
                </div>
                <div className="flex flex-wrap items-stretch gap-3">
                  {OUTPUT_KEYS.filter((k) => prediction[k] != null).map((k, idx) => (
                    <StatCard
                      key={k}
                      label={OUTPUT_LABELS[k]}
                      value={prediction[k]!}
                      unit={OUTPUT_UNITS[k]}
                      decimals={k === 'density' ? 0 : 2}
                      emphasis={idx === 0}
                      index={idx}
                    />
                  ))}
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          <motion.footer
            className="shrink-0 pt-4 text-center text-[clamp(0.65rem,0.6rem+0.15vw,0.72rem)] tracking-wide text-slate-500"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            Client-side only · IS code checks are reference guardrails, not lab
            certification. Validate any mix with proper laboratory testing.
          </motion.footer>
        </div>
      </LayoutGroup>

      <TrainingSidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        epochs={epochs}
        setEpochs={setEpochs}
        lr={lr}
        setLr={setLr}
        valFrac={valFrac}
        setValFrac={setValFrac}
        onOptimizeHyperparams={applyOptimalHyperparams}
        onViewDataset={() => {
          setSidebarOpen(false);
          setDatasetDialogOpen(true);
        }}
        datasetSummary={{
          rows: parsed.X.length,
          outputs: parsed.availableOutputs.length,
        }}
      />

      <DatasetDialog
        open={datasetDialogOpen}
        onClose={() => setDatasetDialogOpen(false)}
        parsed={parsed}
      />
    </div>
  );
}
