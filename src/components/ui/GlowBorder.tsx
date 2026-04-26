import {
  animate,
  motion,
  useMotionTemplate,
  useMotionValue,
} from 'framer-motion';
import { useEffect } from 'react';

interface GlowBorderProps {
  /**
   * Width of the animated ring in px. 1.5 reads as "barely there", 2 is the
   * sweet spot for an emphasized button.
   */
  width?: number;
  /** Seconds for one full rotation. Slower = more premium, faster = more urgent. */
  durationSec?: number;
  className?: string;
}

/**
 * Absolutely-positioned masked conic-gradient that paints ONLY the border ring
 * of the parent. Continuously rotates the gradient angle. Hidden by default —
 * fades in via the parent's `group-hover` state.
 *
 * Implementation notes:
 *   - The "ring-only" effect uses the standard `padding + mask-composite: exclude`
 *     trick: the conic gradient fills the box, and a double mask subtracts the
 *     content-box, leaving only the padding ring.
 *   - The angle is animated through a MotionValue + useMotionTemplate so the
 *     conic-gradient string updates each frame without React re-renders.
 *   - The parent must have `position: relative` and the `group` class.
 */
export function GlowBorder({
  width = 1.5,
  durationSec = 4,
  className = '',
}: GlowBorderProps) {
  const angle = useMotionValue(0);

  useEffect(() => {
    const controls = animate(angle, 360, {
      duration: durationSec,
      ease: 'linear',
      repeat: Infinity,
    });
    return () => controls.stop();
  }, [angle, durationSec]);

  const background = useMotionTemplate`conic-gradient(from ${angle}deg, rgba(45,212,191,0) 0%, rgba(45,212,191,0.95) 22%, rgba(196,232,224,0.55) 44%, rgba(252,211,77,0.7) 60%, rgba(45,212,191,0.95) 78%, rgba(45,212,191,0) 100%)`;

  return (
    <motion.span
      aria-hidden
      style={{
        background,
        padding: width,
        WebkitMask:
          'linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0)',
        WebkitMaskComposite: 'xor',
        maskComposite: 'exclude',
      }}
      className={[
        'pointer-events-none absolute inset-0 rounded-[inherit]',
        'opacity-0 transition-opacity duration-300 ease-out',
        'group-hover:opacity-100 group-focus-visible:opacity-100',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    />
  );
}
