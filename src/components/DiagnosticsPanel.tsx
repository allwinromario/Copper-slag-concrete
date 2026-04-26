import { motion } from 'framer-motion';
import type { TrainedBundle } from '../ann/trainAnn';
import {
  OUTPUT_LABELS,
  OUTPUT_UNITS,
  type BatchProgress,
  type EpochLog,
} from '../ann/types';
import { fadeUp, springSoft, staggerContainer } from '../lib/motion';
import { LossChart, ScatterPlot } from './TrainingCharts';

export interface DiagnosticsPanelProps {
  training: boolean;
  error: string | null;
  epochLogs: EpochLog[];
  bundle: TrainedBundle | null;
  batchProgress?: BatchProgress | null;
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
  batchProgress,
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
          className="mb-3 flex flex-col gap-2 rounded-xl border border-cyan-400/20 bg-cyan-400/[0.05] px-3 py-2.5"
          aria-live="polite"
          aria-atomic="true"
        >
          <div className="flex items-baseline justify-between gap-3">
            <span className="flex items-center gap-2 text-sm font-medium text-cyan-200">
              <span
                className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-cyan-400/30 border-t-cyan-300"
                aria-hidden
              />
              Training network…
            </span>
            {batchProgress && (
              <span className="font-mono text-[0.75rem] tabular-nums text-cyan-100/90">
                {batchProgress.samplesProcessed.toLocaleString()}
                <span className="text-cyan-200/55"> / </span>
                {batchProgress.trainSize.toLocaleString()} rows
              </span>
            )}
          </div>
          {batchProgress && (
            <>
              <div
                className="relative h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={batchProgress.trainSize}
                aria-valuenow={batchProgress.samplesProcessed}
                aria-label="Training rows processed"
              >
                <motion.div
                  className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-cyan-400/80 to-cyan-300"
                  initial={false}
                  animate={{
                    width: `${(batchProgress.samplesProcessed / Math.max(1, batchProgress.trainSize)) * 100}%`,
                  }}
                  transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
                />
              </div>
              <p className="font-mono text-[0.65rem] text-slate-500">
                Epoch {batchProgress.epoch} / {batchProgress.totalEpochs}
              </p>
            </>
          )}
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

        {bundle && bundle.outputs.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={springSoft}
            className="flex flex-col gap-4"
          >
            <p className="font-mono text-[0.7rem] text-slate-500">
              Hold-out: actual vs predicted ·{' '}
              <span className="text-slate-400">
                {bundle.rowsUsed} row{bundle.rowsUsed === 1 ? '' : 's'} used
              </span>
              {bundle.rowsDroppedForNaN > 0 && (
                <span className="text-amber-300/90">
                  {' '}
                  · {bundle.rowsDroppedForNaN} dropped for missing outputs
                </span>
              )}
            </p>
            {bundle.outputs.map((k) => {
              const m = bundle.testMetrics[k];
              if (!m) return null;
              return (
                <div
                  key={k}
                  className="rounded-xl border border-white/5 bg-black/20 p-2"
                >
                  <p className="mb-1 px-1 text-xs font-semibold tracking-wide text-slate-300">
                    {OUTPUT_LABELS[k]}{' '}
                    <span className="text-slate-500">({OUTPUT_UNITS[k]})</span>
                  </p>
                  <ScatterPlot
                    actual={m.actuals}
                    pred={m.predictions}
                    unit={OUTPUT_UNITS[k]}
                    title={OUTPUT_LABELS[k]}
                  />
                  <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 px-1 font-mono text-[0.7rem] text-emerald-300/95">
                    <li>R² = {m.r2.toFixed(4)}</li>
                    <li>
                      RMSE = {m.rmse.toFixed(3)} {OUTPUT_UNITS[k]}
                    </li>
                    <li>
                      MAE = {m.mae.toFixed(3)} {OUTPUT_UNITS[k]}
                    </li>
                  </ul>
                </div>
              );
            })}
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
