import { AnimatePresence, motion } from 'framer-motion';
import { useEffect } from 'react';

interface MethodologyDialogProps {
  open: boolean;
  onClose: () => void;
}

interface Clause {
  code: string;
  title: string;
  body: string;
  formula?: string;
}

const CLAUSES: Clause[] = [
  {
    code: 'IS 456:2000 — Table 5',
    title: 'Max w/c, min cement, min grade vs. exposure',
    body:
      'For each exposure class (Mild → Extreme), Table 5 caps the free water/cement ratio and floors the cement content and the characteristic strength. The IS check panel uses these limits literally.',
    formula:
      'Mild 0.55 / 300 / M20 · Moderate 0.50 / 300 / M25 · Severe 0.45 / 320 / M30 · Very Severe 0.45 / 340 / M35 · Extreme 0.40 / 360 / M40',
  },
  {
    code: 'IS 456:2000 — cl. 6.2.2',
    title: 'Flexural tensile strength from fck',
    body:
      'When flexural strength is not measured, IS 456 lets you estimate it from the cube compressive strength.',
    formula: 'f_cr = 0.7 · √fck   (MPa)',
  },
  {
    code: 'IS 456:2000 — cl. 6.2.3.1',
    title: 'Short-term static modulus of elasticity',
    body:
      'Used as a sanity reference next to the ANN-predicted Ec. Assumes normal-weight concrete.',
    formula: 'E_c = 5000 · √fck   (MPa)   →   ≈ 5·√fck   (GPa)',
  },
  {
    code: 'IS 10262:2019 — cl. 3.2',
    title: 'Target mean strength for mix design',
    body:
      'The mix is designed for a target mean strength higher than the characteristic strength to account for variability.',
    formula:
      'f_target = fck + 1.65·σ\nσ from IS 10262 Table 1: 3.5 (≤M15) · 4.0 (M20–M25) · 5.0 (≥M30)',
  },
  {
    code: 'IS 383:2016 — Table 4',
    title: 'Fineness-modulus zones for fine aggregate',
    body:
      'The check classifies the input fineness modulus into Zones I–IV. A value outside these ranges is flagged so that the mix is not silently accepted.',
    formula:
      'Zone I 2.71–3.41 · Zone II 2.41–3.10 · Zone III 2.11–2.80 · Zone IV 1.80–2.40',
  },
  {
    code: 'Copper-slag practical bound',
    title: 'Replacement of fine aggregate by copper slag',
    body:
      'Not an IS clause: published studies (Mithun & Narasimhan 2016, Ambily et al. 2015, Khanzadi & Behnood 2009) consistently report strength gains up to ~40–50% replacement and losses beyond that. The optimizer caps replacement at 50%.',
    formula: 'copper slag % ∈ [0, 50]',
  },
];

const ROLE = `These clauses are evaluated as guardrails — they validate the inputs you type and constrain the optimizer's search to IS-compliant mixes. They are NOT injected into the ANN's loss function: the network learns the input → output mapping from the dataset alone, and the IS values are shown alongside its predictions for context.`;

export function MethodologyDialog({ open, onClose }: MethodologyDialogProps) {
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
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4 py-8 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          role="dialog"
          aria-modal="true"
          aria-label="IS code methodology"
        >
          <motion.div
            className="relative max-h-full w-full max-w-2xl overflow-y-auto rounded-2xl border border-white/10 bg-slate-950/95 p-6 shadow-2xl shadow-black/60"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ type: 'spring', stiffness: 320, damping: 28 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <p className="text-[0.6rem] font-semibold uppercase tracking-[0.16em] text-amber-200/90">
                  Methodology
                </p>
                <h2 className="mt-1 text-lg font-semibold text-white">
                  Indian Standard references used in this app
                </h2>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close"
                className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-xs font-medium text-slate-300 transition-colors hover:border-white/30 hover:text-white"
              >
                Close
              </button>
            </div>

            <p className="mb-4 rounded-xl border border-amber-200/15 bg-amber-200/[0.04] p-3 text-xs leading-relaxed text-amber-100/85">
              {ROLE}
            </p>

            <div className="flex flex-col gap-3">
              {CLAUSES.map((c) => (
                <div
                  key={c.code}
                  className="rounded-xl border border-white/10 bg-black/30 p-3"
                >
                  <p className="text-[0.6rem] font-semibold uppercase tracking-[0.14em] text-cyan-300/90">
                    {c.code}
                  </p>
                  <p className="mt-0.5 text-sm font-semibold text-white">
                    {c.title}
                  </p>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-300">
                    {c.body}
                  </p>
                  {c.formula && (
                    <pre className="mt-2 whitespace-pre-wrap break-words rounded-md bg-black/50 p-2 font-mono text-[0.7rem] text-emerald-200/90">
                      {c.formula}
                    </pre>
                  )}
                </div>
              ))}
            </div>

            <p className="mt-4 text-[0.65rem] leading-relaxed text-slate-500">
              Citations are paraphrased for brevity. Refer to the published IS
              codes for the exact text and any errata.
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
