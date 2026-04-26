import {
  forwardRef,
  type InputHTMLAttributes,
  type ReactNode,
} from 'react';

interface FieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: ReactNode;
  hint?: ReactNode;
  /** Optional inline trailing badge / unit */
  trailing?: ReactNode;
}

/**
 * Form input with a small uppercase label and a soft focus ring.
 * Designed to compose into a column or grid; the wrapper is `flex flex-col`.
 */
export const Field = forwardRef<HTMLInputElement, FieldProps>(function Field(
  { label, hint, trailing, className = '', id, ...rest },
  ref
) {
  const inputId =
    id ?? `f-${Math.random().toString(36).slice(2, 9)}`;
  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={inputId}
        className="text-[0.65rem] font-medium uppercase tracking-[0.1em] text-slate-500"
      >
        {label}
      </label>
      <div className="relative">
        <input
          ref={ref}
          id={inputId}
          className={[
            'w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2.5 font-mono text-sm text-white',
            'placeholder:text-slate-600',
            'transition-[border-color,box-shadow,background-color] duration-200',
            'hover:border-white/20',
            'focus:border-cyan-400/60 focus:outline-none focus:ring-2 focus:ring-cyan-400/25',
            'disabled:cursor-not-allowed disabled:opacity-50',
            trailing ? 'pr-10' : '',
            className,
          ]
            .filter(Boolean)
            .join(' ')}
          {...rest}
        />
        {trailing && (
          <span className="pointer-events-none absolute inset-y-0 right-3 flex items-center font-mono text-[0.65rem] text-slate-500">
            {trailing}
          </span>
        )}
      </div>
      {hint && (
        <p className="text-[0.65rem] leading-relaxed text-slate-500">{hint}</p>
      )}
    </div>
  );
});
