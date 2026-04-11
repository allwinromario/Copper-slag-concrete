import { AnimatePresence, motion, useReducedMotion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { durationEase, springSoft, staggerContainer, tapScale } from '../lib/motion';
import { MODEL_ARCHITECTURE } from '../ann/trainAnn';

export interface TrainingSidebarProps {
  open: boolean;
  onClose: () => void;
  epochs: number;
  setEpochs: (n: number) => void;
  lr: number;
  setLr: (n: number) => void;
  valFrac: number;
  setValFrac: (n: number) => void;
  onOptimizeHyperparams: () => void;
}

const sectionItem = {
  hidden: { opacity: 0, x: 16 },
  show: (i: number) => ({
    opacity: 1,
    x: 0,
    transition: { ...springSoft, delay: i * 0.05 },
  }),
};

export function TrainingSidebar({
  open,
  onClose,
  epochs,
  setEpochs,
  lr,
  setLr,
  valFrac,
  setValFrac,
  onOptimizeHyperparams,
}: TrainingSidebarProps) {
  const reduce = useReducedMotion();
  const scrollRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ container: scrollRef });
  const headerShadow = useTransform(
    scrollYProgress,
    [0, 0.12],
    ['0 0 0 rgba(0,0,0,0)', '0 12px 40px -12px rgba(0,0,0,0.5)']
  );

  const transition = reduce ? { duration: 0.2 } : springSoft;

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="sb-backdrop"
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={durationEase}
            aria-hidden
            onClick={onClose}
          />
          <motion.aside
            key="sb-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="sidebar-title"
            className="fixed right-0 top-0 z-50 flex h-[100dvh] w-[min(100vw-0.75rem,clamp(380px,32vw,520px))] max-w-full flex-col border-l border-amber-200/15 bg-gradient-to-b from-[#10121a]/98 to-[#07080c]/99 shadow-[-16px_0_64px_rgba(0,0,0,0.55)] backdrop-blur-2xl"
            initial={{ x: '105%' }}
            animate={{ x: 0 }}
            exit={{ x: '105%' }}
            transition={transition}
          >
            <motion.div
              className="flex shrink-0 items-center justify-between border-b border-white/10 px-5 py-4"
              style={{ boxShadow: headerShadow }}
            >
              <h2
                id="sidebar-title"
                className="text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-amber-200/90"
              >
                Training setup
              </h2>
              <motion.button
                type="button"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 transition-colors hover:border-white/20 hover:bg-white/10 hover:text-white"
                onClick={onClose}
                aria-label="Close training panel"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <span aria-hidden className="text-lg leading-none">
                  ×
                </span>
              </motion.button>
            </motion.div>

            <motion.div
              ref={scrollRef}
              className="min-h-0 flex-1 overflow-y-auto scroll-smooth px-5 pb-10 pt-4"
              variants={staggerContainer}
              initial="hidden"
              animate="show"
            >
              <motion.section custom={0} variants={sectionItem} className="mb-8">
                <h3 className="mb-2 text-sm font-semibold text-white">Model architecture</h3>
                <p className="mb-3 font-mono text-xs text-slate-500">
                  Optimizer: Adam · Loss: MSE · Target z-score normalized
                </p>
                <div className="overflow-hidden rounded-xl border border-white/10 bg-black/30">
                  <table className="w-full border-collapse text-sm">
                    <thead>
                      <tr className="border-b border-white/10 text-left text-[0.65rem] uppercase tracking-wide text-slate-500">
                        <th className="px-3 py-2">Layer</th>
                        <th className="px-3 py-2">Units</th>
                        <th className="px-3 py-2">Act.</th>
                        <th className="px-3 py-2">Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {MODEL_ARCHITECTURE.map((row, idx) => (
                        <motion.tr
                          key={`${row.layer}-${idx}`}
                          initial={{ opacity: 0, x: -8 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.03, ...springSoft }}
                          className="border-b border-white/5 text-slate-400 last:border-0 hover:bg-white/[0.03]"
                        >
                          <td className="px-3 py-2 font-medium text-slate-200">{row.layer}</td>
                          <td className="px-3 py-2">
                            {'units' in row ? row.units : row.shape}
                          </td>
                          <td className="px-3 py-2">{row.activation}</td>
                          <td className="px-3 py-2 text-xs">{row.notes}</td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.section>

              <motion.section custom={1} variants={sectionItem}>
                <h3 className="mb-2 text-sm font-semibold text-white">Hyperparameters</h3>
                <p className="mb-3 text-sm text-slate-500">
                  Used when you click <strong className="text-slate-300">Train ANN</strong> on the
                  main workspace.
                </p>
                <motion.button
                  type="button"
                  className="mb-4 w-full rounded-xl border border-amber-200/25 bg-amber-200/[0.08] px-4 py-2.5 text-sm font-semibold text-amber-100"
                  onClick={onOptimizeHyperparams}
                  title="Apply tuned epochs, learning rate, and validation split for this demo"
                  whileHover={{ scale: 1.01, borderColor: 'rgba(253,230,138,0.45)' }}
                  whileTap={tapScale}
                >
                  Optimize hyperparameters
                </motion.button>
                <div className="flex flex-wrap gap-4">
                  <label className="flex flex-col gap-1 text-xs font-medium text-slate-500">
                    Epochs
                    <input
                      id="ep"
                      type="number"
                      min={10}
                      max={2000}
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value))}
                      className="w-28 rounded-lg border border-white/10 bg-black/40 px-3 py-2 font-mono text-sm text-white focus:border-cyan-500/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-xs font-medium text-slate-500">
                    Learn rate
                    <input
                      id="lr-in"
                      type="number"
                      step={0.005}
                      min={0.0001}
                      max={0.2}
                      value={lr}
                      onChange={(e) => setLr(Number(e.target.value))}
                      className="w-28 rounded-lg border border-white/10 bg-black/40 px-3 py-2 font-mono text-sm text-white focus:border-cyan-500/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-xs font-medium text-slate-500">
                    Val. fraction
                    <input
                      id="vf"
                      type="number"
                      step={0.05}
                      min={0.1}
                      max={0.4}
                      value={valFrac}
                      onChange={(e) => setValFrac(Number(e.target.value))}
                      className="w-28 rounded-lg border border-white/10 bg-black/40 px-3 py-2 font-mono text-sm text-white focus:border-cyan-500/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/30"
                    />
                  </label>
                </div>
              </motion.section>
            </motion.div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
