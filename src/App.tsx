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
import {
  buildOptimizedSyntheticCsv,
  OPTIMAL_TRAINING,
  searchMaxStrengthInputs,
  searchMaxStrengthSynthetic,
} from './ann/optimize';
import { predictStrength, trainAnn, type TrainedBundle } from './ann/trainAnn';
import type { EpochLog, FeatureRow } from './ann/types';
import { FEATURE_LABELS } from './ann/types';
import { DiagnosticsPanel } from './components/DiagnosticsPanel';
import { TrainingSidebar } from './components/TrainingSidebar';
import { loadStoredFeatureInputs, persistFeatureInputs } from './lib/featureInputsStorage';
import {
  fadeUp,
  springLayout,
  springSoft,
  staggerContainer,
  tapScale,
} from './lib/motion';

type DataSource = 'sample' | 'sem';

const defaultInputs: FeatureRow = [320, 165, 20, 28, 7.2, 2.1];

const TABLE_HEADERS = [
  'Cement (kg/m³)',
  'Water (kg/m³)',
  'Slag %',
  'Days',
  'Por. %',
  'Crack (mm/mm²)',
  "f'c (MPa)",
] as const;

const glassPanel =
  'relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-white/[0.07] to-white/[0.02] shadow-2xl shadow-black/50 backdrop-blur-2xl ring-1 ring-cyan-400/5 transition-shadow duration-500 hover:shadow-cyan-500/10 hover:ring-cyan-400/15';

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

  return (
    <>
      <motion.div
        className="flex shrink-0 items-center justify-between gap-3 border-b border-white/10 bg-amber-200/[0.06] px-3 py-2"
        style={{ opacity: headOpacity }}
      >
        <h3 className="text-[0.65rem] font-semibold uppercase tracking-[0.1em] text-amber-200/90">
          Dataset (CSV)
        </h3>
        <span className="font-mono text-[0.65rem] text-slate-500">
          {total} row{total === 1 ? '' : 's'} · scroll
        </span>
      </motion.div>
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-auto overscroll-contain [-webkit-overflow-scrolling:touch]"
      >
        <table className="w-full border-collapse font-mono text-[0.7rem]">
          <thead>
            <tr className="sticky top-0 z-[1] border-b border-amber-200/15 bg-[#0c0e14]/95 backdrop-blur-md">
              <th className="w-10 px-2 py-2 text-center text-[0.65rem] font-semibold uppercase tracking-wide text-slate-500">
                #
              </th>
              {TABLE_HEADERS.map((h) => (
                <th
                  key={h}
                  scope="col"
                  title={h}
                  className="whitespace-nowrap px-2 py-2 text-right text-[0.65rem] font-semibold text-slate-500"
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
                className="border-b border-white/[0.06] odd:bg-white/[0.02] hover:bg-cyan-400/[0.04]"
              >
                <td className="px-2 py-1.5 text-center text-slate-500">{i + 1}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[0].toFixed(1)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[1].toFixed(1)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[2].toFixed(1)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[3]}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[4].toFixed(2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{row[5].toFixed(2)}</td>
                <td className="px-2 py-1.5 text-right text-slate-200">{parsed.y[i].toFixed(2)}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function renderDataColumn(
  dataSource: DataSource,
  syntheticRowCount: number,
  setSyntheticRowCount: (n: number) => void,
  loadSynthetic: () => void,
  setCsvText: (s: string) => void,
  setError: (e: string | null) => void,
  parsed: ReturnType<typeof parseDataset>,
  markDatasetPrimed: () => void,
  onOptimizeSetup: () => void
) {
  return (
    <>
      <p className="mb-1 text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-amber-200/90">
        Data source
      </p>
      <h2 className="mb-3 text-base font-semibold text-white sm:text-lg">Training data</h2>

      <div className="shrink-0">
        {dataSource === 'sample' && (
          <>
            <p className="mb-3 text-sm leading-relaxed text-slate-400">
              Generate data or edit CSV in <strong className="text-slate-200">Training setup</strong>
              .
            </p>
            <div className="mb-3 flex flex-wrap items-end gap-3">
              <label className="flex max-w-[6rem] flex-col gap-1 text-xs font-medium text-slate-500">
                Row count
                <input
                  type="number"
                  min={20}
                  max={500}
                  value={syntheticRowCount}
                  onChange={(e) => setSyntheticRowCount(Number(e.target.value))}
                  className="rounded-lg border border-white/10 bg-black/40 px-3 py-2 font-mono text-sm text-white focus:border-cyan-500/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
                />
              </label>
              <motion.button
                type="button"
                className="rounded-xl bg-gradient-to-br from-cyan-400 to-teal-600 px-4 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/20"
                onClick={() => {
                  loadSynthetic();
                  markDatasetPrimed();
                }}
                whileHover={{ scale: 1.03, boxShadow: '0 0 32px -4px rgba(62,232,214,0.45)' }}
                whileTap={tapScale}
              >
                Generate plausible data
              </motion.button>
              <motion.button
                type="button"
                className="rounded-xl border border-white/15 bg-white/5 px-4 py-2.5 text-sm font-semibold text-slate-200"
                onClick={() => {
                  setCsvText(DEMO_CSV);
                  setError(null);
                  markDatasetPrimed();
                }}
                whileHover={{ scale: 1.02, borderColor: 'rgba(232,212,184,0.35)' }}
                whileTap={tapScale}
              >
                Load small demo
              </motion.button>
              <motion.button
                type="button"
                className="rounded-xl border border-amber-200/25 bg-amber-200/[0.08] px-4 py-2.5 text-sm font-semibold text-amber-100"
                onClick={onOptimizeSetup}
                title="Larger synthetic set + tuned epochs, learning rate, and validation split"
                whileHover={{ scale: 1.02, borderColor: 'rgba(253,230,138,0.45)' }}
                whileTap={tapScale}
              >
                Optimize setup
              </motion.button>
            </div>
            <div className="mb-2 flex flex-wrap gap-2">
              {parsed.error ? (
                <span className="inline-flex rounded-full border border-red-400/25 bg-red-500/10 px-3 py-1 font-mono text-xs text-red-300">
                  {parsed.error}
                </span>
              ) : (
                <span className="inline-flex rounded-full border border-emerald-400/25 bg-emerald-500/10 px-3 py-1 font-mono text-xs text-emerald-300">
                  {parsed.X.length} row{parsed.X.length === 1 ? '' : 's'} in dataset
                </span>
              )}
            </div>
          </>
        )}

        {dataSource === 'sem' && (
          <div className="mb-3 rounded-xl border border-dashed border-amber-200/20 bg-amber-200/[0.05] p-4 text-sm leading-relaxed text-slate-400">
            <strong className="text-amber-200/90">Future work.</strong> SEM upload and
            auto-extraction are not implemented. Use ImageJ offline, then paste CSV in{' '}
            <strong className="text-slate-200">Training setup</strong>. The table below reflects
            whatever is currently loaded.
          </div>
        )}
      </div>

      <div className="mt-1 flex min-h-0 flex-1 flex-col">
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-white/10 bg-black/40">
          <DatasetPreviewTable parsed={parsed} />
        </div>
      </div>
    </>
  );
}

export default function App() {
  const reduce = useReducedMotion();
  const [mqWide, setMqWide] = useState(true);

  const [dataSource, setDataSource] = useState<DataSource>('sample');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [csvText, setCsvText] = useState(DEMO_CSV);
  const [syntheticRowCount, setSyntheticRowCount] = useState(100);
  const [epochs, setEpochs] = useState(220);
  const [lr, setLr] = useState(0.012);
  const [valFrac, setValFrac] = useState(0.2);
  const [epochLogs, setEpochLogs] = useState<EpochLog[]>([]);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<TrainedBundle | null>(null);
  const [inputs, setInputs] = useState<FeatureRow>(() =>
    loadStoredFeatureInputs([...defaultInputs])
  );
  const [prediction, setPrediction] = useState<number | null>(null);

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

  const markDatasetPrimed = useCallback(() => {
    setDatasetPrimed(true);
    setHasSavedInputs(false);
  }, []);

  const loadSynthetic = useCallback(() => {
    const n = Math.min(500, Math.max(20, Math.round(syntheticRowCount)));
    const { X, y } = generateDemoData(n);
    const header =
      'Cement,Water,Copper slag %,Curing days,Porosity %,Crack density,Compressive strength';
    const rows = X.map(
      (r, i) =>
        `${r[0].toFixed(1)},${r[1].toFixed(1)},${r[2].toFixed(1)},${r[3]},${r[4].toFixed(2)},${r[5].toFixed(2)},${y[i]}`
    );
    setCsvText([header, ...rows].join('\n'));
    setError(null);
  }, [syntheticRowCount]);

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
    setBundle(null);
    setPrediction(null);

    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }

    try {
      const result = await trainAnn(parsed.X, parsed.y, {
        epochs,
        learningRate: lr,
        validationFraction: valFrac,
        onEpoch: (log) => {
          setEpochLogs((prev) => [...prev, log]);
        },
      });
      modelRef.current = result.model;
      setBundle(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setTraining(false);
    }
  };

  const handleMainTrain = async () => {
    setThreeColLayout(true);
    await train();
  };

  const runPredict = () => {
    if (!bundle) return;
    const v = predictStrength(bundle.model, bundle.norm, inputs);
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
        ? await searchMaxStrengthInputs(bundle.model, bundle.norm)
        : await searchMaxStrengthSynthetic();
      setInputs(next);
      setHasSavedInputs(false);
    } finally {
      optimizeInputsBusy.current = false;
      setOptimizingInputs(false);
    }
  }, [bundle]);

  const refreshApp = useCallback(() => {
    if (modelRef.current) {
      modelRef.current.dispose();
      modelRef.current = null;
    }
    setSidebarOpen(false);
    setDataSource('sample');
    setCsvText(DEMO_CSV);
    setSyntheticRowCount(100);
    setEpochs(220);
    setLr(0.012);
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
      <div        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_85%_55%_at_0%_-10%,rgba(212,184,150,0.14),transparent_52%),radial-gradient(ellipse_70%_45%_at_100%_0%,rgba(45,212,191,0.08),transparent_48%),radial-gradient(ellipse_60%_40%_at_50%_110%,rgba(15,118,110,0.12),transparent_55%)]"
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
            className="mb-4 flex shrink-0 flex-wrap items-start justify-between gap-4 border-b border-white/10 pb-4"
            style={{ y: heroY }}
          >
            <header className="min-w-0 flex-1">
              <motion.h1
                className="mb-2 bg-gradient-to-r from-white via-white to-amber-200/90 bg-clip-text font-display text-[clamp(1.5rem,1.35rem+2.2vw,3rem)] font-semibold leading-tight tracking-[0.03em] text-transparent"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={reduce ? { duration: 0.2 } : springSoft}
              >
                Copper slag concrete
              </motion.h1>
              <p className="max-w-[min(42rem,55vw)] text-[clamp(0.8125rem,0.75rem+0.35vw,0.9375rem)] leading-relaxed text-slate-400">
                <strong className="font-semibold text-amber-200/90">ANN</strong> compressive strength
                · TensorFlow.js in-browser. Mix design plus SEM-derived{' '}
                <strong className="text-slate-200">porosity</strong> and{' '}
                <strong className="text-slate-200">crack density</strong> as inputs.
              </p>
            </header>
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              <motion.button
                type="button"
                className="flex items-center gap-2 rounded-2xl border border-white/15 bg-white/[0.06] px-4 py-2.5 text-sm font-semibold tracking-wide text-slate-100 shadow-xl shadow-black/30 backdrop-blur-xl"
                onClick={refreshApp}
                aria-label="Reset application — clears model and training data; inference mix values stay and remain saved in the browser"
                title="Reset model and data (inference numbers stay saved in this browser)"
                whileHover={{
                  scale: 1.03,
                  borderColor: 'rgba(255,255,255,0.28)',
                  boxShadow: '0 0 28px -8px rgba(255,255,255,0.12)',
                }}
                whileTap={tapScale}
              >
                <svg
                  className="h-4 w-4 shrink-0 text-slate-300"
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
              </motion.button>
              <motion.button
                type="button"
                className="flex shrink-0 items-center gap-2 rounded-2xl border border-white/15 bg-white/[0.06] px-5 py-2.5 text-sm font-semibold tracking-wide text-slate-100 shadow-xl shadow-black/30 backdrop-blur-xl"
                onClick={() => setSidebarOpen(true)}
                aria-expanded={sidebarOpen}
                whileHover={{
                  scale: 1.03,
                  borderColor: 'rgba(62,232,214,0.35)',
                  boxShadow: '0 0 40px -8px rgba(62,232,214,0.25)',
                }}
                whileTap={tapScale}
              >
                <span
                  className="h-2 w-2 shrink-0 rounded-full bg-cyan-400 shadow-[0_0_14px_rgba(62,232,214,0.55)]"
                  aria-hidden
                />
                Training setup
              </motion.button>
            </div>
          </motion.div>

          <motion.section
            className={`${glassPanel} mb-4 shrink-0 px-[clamp(1rem,2.5vw,1.5rem)] py-[clamp(0.85rem,2vw,1.15rem)]`}
            variants={staggerContainer}
            initial="hidden"
            animate="show"
            aria-label="Data source selection"
          >
            <motion.p
              variants={fadeUp}
              className="mb-1 text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-amber-200/90"
            >
              Data source
            </motion.p>
            <motion.h2 variants={fadeUp} className="mb-4 text-base font-semibold text-white sm:text-lg">
              Build your training table
            </motion.h2>
            <motion.div variants={fadeUp} role="tablist" aria-label="Data source" className="flex gap-1.5 rounded-2xl border border-white/10 bg-black/40 p-1.5 shadow-inner shadow-black/40">
              {(['sample', 'sem'] as const).map((tab) => (
                <motion.button
                  key={tab}
                  type="button"
                  role="tab"
                  aria-selected={dataSource === tab}
                  onClick={() => setDataSource(tab)}
                  className={`relative flex-1 rounded-xl px-4 py-3 text-sm font-semibold tracking-wide transition-colors ${
                    dataSource === tab
                      ? 'text-slate-950'
                      : 'text-slate-500 hover:bg-white/[0.04] hover:text-slate-300'
                  }`}
                  whileTap={{ scale: 0.98 }}
                >
                  {dataSource === tab && (
                    <motion.span
                      layoutId="tab-pill"
                      className="absolute inset-0 -z-10 rounded-xl bg-gradient-to-br from-amber-200 via-amber-400/90 to-amber-700/80 shadow-lg shadow-amber-500/20"
                      transition={springSoft}
                    />
                  )}
                  {tab === 'sample' ? 'Sample data' : 'SEM images'}
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
              {renderDataColumn(
                dataSource,
                syntheticRowCount,
                setSyntheticRowCount,
                loadSynthetic,
                setCsvText,
                setError,
                parsed,
                markDatasetPrimed,
                applyOptimalTrainingSetup
              )}
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
                    className="flex min-h-0 flex-1 flex-col items-stretch justify-center text-center"
                    initial={{ opacity: 0, scale: 0.92, filter: 'blur(8px)' }}
                    animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                    exit={{ opacity: 0, scale: 0.96 }}
                    transition={springSoft}
                  >
                    <p className="mb-1 text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-amber-200/90">
                      Train
                    </p>
                    <h2 className="mb-2 text-base font-semibold text-white">Ready</h2>
                    <p className="mb-6 text-left text-sm leading-relaxed text-slate-400">
                      Dataset loaded and inputs saved. Tune epochs &amp; learning rate in{' '}
                      <strong className="text-slate-200">Training setup</strong> if needed.
                    </p>
                    <motion.button
                      type="button"
                      disabled={training}
                      onClick={handleMainTrain}
                      className="w-full rounded-2xl bg-gradient-to-br from-cyan-400 to-teal-600 py-4 text-sm font-bold tracking-wide text-slate-950 shadow-lg shadow-cyan-500/25 disabled:opacity-40"
                      whileHover={!training ? { scale: 1.02, boxShadow: '0 0 40px -4px rgba(62,232,214,0.45)' } : {}}
                      whileTap={!training ? tapScale : {}}
                    >
                      {training ? (
                        <span className="inline-flex items-center justify-center gap-2">
                          <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-900/30 border-t-slate-900" />
                          Training…
                        </span>
                      ) : (
                        'Train ANN'
                      )}
                    </motion.button>
                  </motion.div>
                )}
                {workspacePhase === 'p3' && (
                  <motion.div
                    key="diag"
                    className="flex min-h-0 flex-1 flex-col gap-3"
                    initial={{ opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -12 }}
                    transition={springSoft}
                  >
                    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-cyan-400/15 bg-gradient-to-b from-cyan-400/[0.06] to-transparent">
                      <DiagnosticsPanel
                        training={training}
                        error={error}
                        epochLogs={epochLogs}
                        bundle={bundle}
                      />
                    </div>
                    <motion.button
                      type="button"
                      className="w-full shrink-0 rounded-2xl bg-gradient-to-br from-cyan-400 to-teal-600 py-3.5 text-sm font-bold text-slate-950 shadow-lg shadow-cyan-500/20 disabled:opacity-40"
                      disabled={!bundle || training}
                      onClick={runPredict}
                      whileHover={bundle && !training ? { scale: 1.02 } : {}}
                      whileTap={bundle && !training ? tapScale : {}}
                    >
                      Generate predicted strength
                    </motion.button>
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
              <p className="mb-1 text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-amber-200/90">
                Inference
              </p>
              <h2 className="mb-3 text-base font-semibold text-white sm:text-lg">
                Predict compressive strength
              </h2>
              <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain pr-0.5">
                {threeColLayout && (
                  <p className="mb-4 text-sm leading-relaxed text-slate-400">
                    Adjust inputs on the right, then use Generate predicted strength in Diagnostics — the score appears below.
                  </p>
                )}
                <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {FEATURE_LABELS.map((label, i) => (
                    <motion.label
                      key={label}
                      className="flex flex-col gap-1.5 text-xs font-medium text-slate-500"
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.04, ...springSoft }}
                    >
                      {label}
                      <input
                        type="number"
                        step="any"
                        value={inputs[i]}
                        onChange={(e) => updateInput(i, Number(e.target.value))}
                        className="rounded-xl border border-white/10 bg-black/40 px-3 py-2.5 font-mono text-sm text-white transition-shadow focus:border-cyan-500/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
                      />
                    </motion.label>
                  ))}
                </div>
                <div className="mt-auto flex flex-wrap items-center gap-2 pt-1">
                  <motion.button
                    type="button"
                    className="rounded-xl border border-amber-200/25 bg-amber-200/[0.08] px-4 py-2.5 text-sm font-semibold text-amber-100 disabled:opacity-35"
                    disabled={training || optimizingInputs}
                    onClick={() => void applyOptimizeInferenceInputs()}
                    title={
                      bundle
                        ? 'Search for inputs that maximize ANN predicted strength'
                        : 'Suggest a high-strength mix using the demo strength model (train the ANN for neural predictions)'
                    }
                    whileHover={
                      !training && !optimizingInputs
                        ? { scale: 1.02, borderColor: 'rgba(253,230,138,0.45)' }
                        : {}
                    }
                    whileTap={!training && !optimizingInputs ? tapScale : {}}
                  >
                    {optimizingInputs ? (
                      <span className="inline-flex items-center gap-2">
                        <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-amber-200/30 border-t-amber-200" />
                        Optimizing…
                      </span>
                    ) : (
                      'Optimize inputs'
                    )}
                  </motion.button>
                  <motion.button
                    type="button"
                    className="rounded-xl border border-white/15 bg-white/5 px-4 py-2.5 text-sm font-semibold text-slate-200"
                    onClick={saveInputs}
                    whileHover={{ scale: 1.02, borderColor: 'rgba(232,212,184,0.35)' }}
                    whileTap={tapScale}
                  >
                    Save inputs
                  </motion.button>
                  <AnimatePresence>
                    {hasSavedInputs && (
                      <motion.span
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="inline-flex rounded-full border border-emerald-400/25 bg-emerald-500/10 px-3 py-1 font-mono text-xs text-emerald-300"
                      >
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
                className={`${glassPanel} relative mt-2 shrink-0 px-3 py-3 sm:px-4 sm:py-3.5`}
                aria-label={`Predicted compressive strength ${prediction.toFixed(2)} megapascals`}
                aria-live="polite"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 8 }}
                transition={springSoft}
              >
                <motion.button
                  type="button"
                  className="absolute right-2 top-2 z-10 flex h-8 w-8 items-center justify-center rounded-xl border border-white/12 bg-black/35 text-slate-400 backdrop-blur-sm transition-colors hover:border-white/20 hover:text-slate-200"
                  onClick={() => setPrediction(null)}
                  aria-label="Dismiss prediction"
                  whileHover={{ scale: 1.05 }}
                  whileTap={tapScale}
                >
                  <span className="text-lg leading-none" aria-hidden>
                    ×
                  </span>
                </motion.button>
                <motion.div
                  className="flex flex-col items-center justify-center gap-1 px-6 text-center"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.04, ...springSoft }}
                >
                  <p className="text-sm font-medium text-slate-400 sm:text-base">
                    compressive strength (MPa)
                  </p>
                  <p className="font-mono text-xl font-semibold tabular-nums tracking-tight text-cyan-300 sm:text-2xl">
                    <span>{prediction.toFixed(2)}</span>
                    <span className="ml-1.5 text-base font-medium text-slate-500 sm:text-lg">MPa</span>
                  </p>
                </motion.div>
              </motion.section>
            )}
          </AnimatePresence>

          <motion.footer
            className="shrink-0 pt-3 text-center font-mono text-[clamp(0.62rem,0.55rem+0.2vw,0.72rem)] tracking-wider text-slate-500"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            Client-side only — use lab + ImageJ-derived porosity & crack density for real work.
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
      />
    </div>
  );
}
