import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

interface SectionHeaderProps {
  eyebrow?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  className?: string;
}

/**
 * Consistent panel header: small uppercase eyebrow + bold title + optional
 * one-line subtitle and a right-aligned action slot.
 */
export function SectionHeader({
  eyebrow,
  title,
  subtitle,
  action,
  className = '',
}: SectionHeaderProps) {
  return (
    <div
      className={[
        'mb-3 flex items-start justify-between gap-3',
        className,
      ].join(' ')}
    >
      <div className="min-w-0">
        {eyebrow && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            className="mb-1 text-[0.6rem] font-semibold uppercase tracking-[0.14em] text-slate-500"
          >
            {eyebrow}
          </motion.p>
        )}
        <motion.h2
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1], delay: 0.04 }}
          className="text-base font-semibold tracking-tight text-white sm:text-lg"
        >
          {title}
        </motion.h2>
        {subtitle && (
          <p className="mt-1 text-[0.8rem] leading-relaxed text-slate-400">
            {subtitle}
          </p>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}
