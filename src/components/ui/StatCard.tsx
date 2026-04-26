import { motion } from 'framer-motion';
import { CountUp } from './CountUp';

interface StatCardProps {
  label: string;
  value: number;
  unit?: string;
  decimals?: number;
  /** First (primary) card gets the accent treatment + larger numbers. */
  emphasis?: boolean;
  /** 0-based index for stagger timing. */
  index?: number;
}

export function StatCard({
  label,
  value,
  unit,
  decimals = 2,
  emphasis = false,
  index = 0,
}: StatCardProps) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.4,
        delay: index * 0.05,
        ease: [0.22, 1, 0.36, 1],
      }}
      whileHover={{ y: -2 }}
      className={[
        'group relative flex min-w-[8.5rem] flex-1 flex-col items-start gap-1 rounded-2xl border px-4 py-3',
        'transition-[border-color,background-color,box-shadow] duration-300',
        emphasis
          ? 'border-cyan-400/30 bg-gradient-to-b from-cyan-400/[0.08] to-cyan-400/[0.02] shadow-[0_8px_28px_-12px_rgba(45,212,191,0.35)] hover:border-cyan-400/45 hover:shadow-[0_16px_40px_-16px_rgba(45,212,191,0.5)]'
          : 'border-white/10 bg-white/[0.025] hover:border-white/20 hover:bg-white/[0.04]',
      ].join(' ')}
    >
      <p className="text-[0.62rem] font-medium uppercase tracking-[0.12em] text-slate-500">
        {label}
      </p>
      <p
        className={[
          'flex items-baseline gap-1.5 font-mono font-semibold tabular-nums tracking-tight',
          emphasis
            ? 'text-2xl text-cyan-100 sm:text-[1.75rem]'
            : 'text-lg text-slate-100',
        ].join(' ')}
      >
        <CountUp value={value} decimals={decimals} />
        {unit && (
          <span
            className={[
              'text-[0.7rem] font-medium tracking-wide',
              emphasis ? 'text-cyan-300/80' : 'text-slate-500',
            ].join(' ')}
          >
            {unit}
          </span>
        )}
      </p>
    </motion.div>
  );
}
