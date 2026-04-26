import { motion, type HTMLMotionProps } from 'framer-motion';
import { forwardRef, type ReactNode } from 'react';
import { GlowBorder } from './GlowBorder';

type Variant = 'primary' | 'secondary' | 'accent' | 'ghost';
type Size = 'sm' | 'md' | 'lg';

interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  children: ReactNode;
  fullWidth?: boolean;
  /**
   * Reserved for *important* actions only (e.g. Generate prediction, Optimize
   * setup). Adds an animated conic-gradient border ring + a stronger halo
   * shadow on hover. Avoid using on routine buttons — it loses meaning.
   */
  glow?: boolean;
}

const VARIANT: Record<Variant, string> = {
  primary:
    'bg-gradient-to-br from-cyan-300 to-teal-500 text-slate-950 shadow-[0_8px_24px_-12px_rgba(20,184,166,0.55)] hover:shadow-[0_16px_36px_-12px_rgba(45,212,191,0.55)]',
  secondary:
    'border border-white/10 bg-white/[0.04] text-slate-100 hover:border-white/20 hover:bg-white/[0.07]',
  accent:
    'border border-amber-300/25 bg-amber-300/[0.07] text-amber-100 hover:border-amber-300/45 hover:bg-amber-300/[0.1]',
  ghost: 'text-slate-300 hover:bg-white/[0.05] hover:text-white',
};

const SIZE: Record<Size, string> = {
  sm: 'h-8 px-3 text-xs',
  md: 'h-10 px-4 text-sm',
  lg: 'h-12 px-5 text-sm',
};

const Spinner = ({ tone }: { tone: 'dark' | 'light' }) => (
  <span
    className={`h-3.5 w-3.5 animate-spin rounded-full border-2 ${
      tone === 'dark'
        ? 'border-slate-900/30 border-t-slate-900'
        : 'border-white/30 border-t-white'
    }`}
    aria-hidden
  />
);

/** Extra hover halo for glow-enabled buttons; tuned per variant. */
const GLOW_HALO: Record<Variant, string> = {
  primary:
    'hover:shadow-[0_0_0_1px_rgba(45,212,191,0.25),0_18px_44px_-14px_rgba(45,212,191,0.55)]',
  accent:
    'hover:shadow-[0_0_0_1px_rgba(252,211,77,0.25),0_18px_44px_-14px_rgba(252,211,77,0.45)]',
  secondary:
    'hover:shadow-[0_0_0_1px_rgba(255,255,255,0.12),0_16px_36px_-14px_rgba(255,255,255,0.18)]',
  ghost:
    'hover:shadow-[0_0_0_1px_rgba(255,255,255,0.08),0_12px_28px_-12px_rgba(255,255,255,0.12)]',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  function Button(
    {
      variant = 'secondary',
      size = 'md',
      loading = false,
      children,
      className = '',
      disabled,
      fullWidth,
      glow = false,
      ...rest
    },
    ref
  ) {
    const isDisabled = disabled || loading;
    return (
      <motion.button
        ref={ref}
        type={rest.type ?? 'button'}
        disabled={isDisabled}
        whileHover={isDisabled ? undefined : { scale: 1.02 }}
        whileTap={isDisabled ? undefined : { scale: 0.98 }}
        transition={{ type: 'spring', stiffness: 480, damping: 32, mass: 0.6 }}
        className={[
          'group relative inline-flex items-center justify-center gap-2 rounded-xl font-semibold tracking-tight transition-[background,border-color,box-shadow,color] duration-300',
          'disabled:cursor-not-allowed disabled:opacity-40',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40 focus-visible:ring-offset-2 focus-visible:ring-offset-[#07080c]',
          VARIANT[variant],
          SIZE[size],
          fullWidth ? 'w-full' : '',
          glow && !isDisabled ? GLOW_HALO[variant] : '',
          className,
        ]
          .filter(Boolean)
          .join(' ')}
        {...rest}
      >
        {glow && !isDisabled && <GlowBorder />}
        {loading ? (
          <Spinner tone={variant === 'primary' ? 'dark' : 'light'} />
        ) : null}
        <span className="relative inline-flex items-center gap-2">{children}</span>
      </motion.button>
    );
  }
);
