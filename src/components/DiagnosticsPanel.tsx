import { motion } from 'framer-motion';
import type { TrainedBundle } from '../ann/trainAnn';
import type { EpochLog } from '../ann/types';
import { fadeUp, springSoft, staggerContainer } from '../lib/motion';
import { LossChart, ScatterPlot } from './TrainingCharts';

export interface DiagnosticsPanelProps {
  training: boolean;
  error: string | null;
  epochLogs: EpochLog[];
  bundle: TrainedBundle | null;
}

const item = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: springSoft },
};

export function DiagnosticsPanel({
  training,
  error,
  epochLogs,
  bundle,
}: DiagnosticsPanelProps) {
  return (
    <motion.div
      className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain px-2 pb-3 pt-1 sm:px-3"
      variants={staggerContainer}
      initial="hidden"
      animate="show"
    >
      <motion.p
        variants={fadeUp}
        className="mb-1 text-[0.65rem] font-semibold uppercase tracking-[0.14em] text-amber-200/90"
      >
        Diagnostics
      </motion.p>
      <motion.h2 variants={fadeUp} className="mb-3 text-base font-semibold text-white">
        Training metrics
      </motion.h2>

      {error && (
        <motion.p
          variants={item}
          className="mb-3 rounded-lg border border-red-400/25 bg-red-500/10 px-3 py-2 font-mono text-xs text-red-300"
          role="alert"
        >
          {error}
        </motion.p>
      )}

      {training && (
        <motion.div
          variants={item}
          className="mb-3 flex items-center gap-2 text-sm text-cyan-300"
          aria-live="polite"
        >
          <span            className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-cyan-400/30 border-t-cyan-400"
            aria-hidden
          />
          <span>Training network…</span>
        </motion.div>
      )}

      <motion.div variants={item} className="flex flex-col gap-4">
        <div>
          <p className="mb-2 flex flex-wrap gap-4 font-mono text-[0.65rem] text-slate-500">
            <span className="inline-flex items-center gap-1.5">
              <i className="h-0.5 w-3 rounded-full bg-cyan-400" /> Train loss
            </span>
            <span className="inline-flex items-center gap-1.5">
              <i className="h-0.5 w-3 rounded-full bg-amber-400" /> Val loss
            </span>
          </p>
          <div className="rounded-xl border border-white/5 bg-black/20 p-2">
            <LossChart logs={epochLogs} />
          </div>
        </div>
        {bundle && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={springSoft}
          >
            <p className="mb-2 font-mono text-[0.7rem] text-slate-500">
              Hold-out: actual vs predicted
            </p>
            <div className="rounded-xl border border-white/5 bg-black/20 p-2">
              <ScatterPlot
                actual={bundle.testMetrics.actuals}
                pred={bundle.testMetrics.predictions}
              />
            </div>
            <ul className="mt-2 space-y-0.5 pl-4 font-mono text-xs text-emerald-300/95">
              <li>R² = {bundle.testMetrics.r2.toFixed(4)}</li>
              <li>RMSE = {bundle.testMetrics.rmse.toFixed(3)}</li>
              <li>MAE = {bundle.testMetrics.mae.toFixed(3)}</li>
            </ul>
          </motion.div>
        )}
      </motion.div>

      {epochLogs.length > 0 && !training && (
        <motion.div variants={item} className="mt-3">
          <details className="rounded-lg border border-white/10 bg-black/25 text-slate-400 [&_summary]:cursor-pointer [&_summary]:px-1 [&_summary]:py-2 [&_summary]:text-sm">
            <summary>Latest epoch</summary>
            <pre className="mt-1 max-h-24 overflow-auto rounded-md border border-white/5 bg-black/40 p-2 font-mono text-[0.68rem] leading-relaxed text-slate-500">
              {JSON.stringify(epochLogs[epochLogs.length - 1], null, 2)}
            </pre>
          </details>
        </motion.div>
      )}
    </motion.div>
  );
}
