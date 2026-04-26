import { animate, motion, useMotionValue, useTransform } from 'framer-motion';
import { useEffect } from 'react';

interface CountUpProps {
  value: number;
  decimals?: number;
  duration?: number;
  className?: string;
}

/**
 * Smoothly tweens a number from its previous render value to the new `value`.
 * Honours `prefers-reduced-motion` indirectly via the global CSS rule that
 * collapses transitions, but also keeps the duration short by default.
 */
export function CountUp({
  value,
  decimals = 2,
  duration = 0.7,
  className,
}: CountUpProps) {
  const mv = useMotionValue(value);
  const display = useTransform(mv, (v) =>
    Number.isFinite(v) ? v.toFixed(decimals) : '—'
  );

  useEffect(() => {
    const controls = animate(mv, value, {
      duration,
      ease: [0.22, 1, 0.36, 1],
    });
    return () => controls.stop();
  }, [value, duration, mv]);

  return <motion.span className={className}>{display}</motion.span>;
}
