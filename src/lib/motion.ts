import type { Transition, Variants } from 'framer-motion';

/** Apple-like smooth curve */
export const easeInOut: [number, number, number, number] = [0.4, 0, 0.2, 1];

export const springSoft: Transition = {
  type: 'spring',
  stiffness: 320,
  damping: 32,
  mass: 0.85,
};

export const springSnappy: Transition = {
  type: 'spring',
  stiffness: 420,
  damping: 28,
  mass: 0.72,
};

export const springLayout: Transition = {
  type: 'spring',
  stiffness: 280,
  damping: 30,
  mass: 0.9,
};

export const durationEase: Transition = {
  duration: 0.55,
  ease: easeInOut,
};

export const staggerContainer: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.07,
      delayChildren: 0.04,
      ease: easeInOut,
    },
  },
};

export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 22, filter: 'blur(8px)' },
  show: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: springSoft,
  },
};

export const fadeScale: Variants = {
  hidden: { opacity: 0, scale: 0.96, y: 12 },
  show: { opacity: 1, scale: 1, y: 0, transition: springSnappy },
};

export const tapScale = { scale: 0.97 };
export const hoverGlow = {
  scale: 1.02,
  boxShadow: '0 0 40px -8px rgba(62, 232, 214, 0.35), 0 20px 50px -20px rgba(0,0,0,0.55)',
};
