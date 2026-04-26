import type { EpochLog } from '../ann/types';

const LOSS_W = 400;
const LOSS_H = 128;
const LOSS_PAD = 10;

function LossChart({ logs }: { logs: EpochLog[] }) {
  if (logs.length === 0) return null;

  const losses = logs.map((l) => l.loss);
  const valLosses = logs
    .map((l) => l.valLoss)
    .filter((v): v is number => v != null && !Number.isNaN(v));
  const maxL = Math.max(...losses, ...(valLosses.length ? valLosses : [0]), 1e-6);
  const minL = Math.min(...losses, ...(valLosses.length ? valLosses : losses));
  const span = maxL - minL || 1;

  const innerW = LOSS_W - 2 * LOSS_PAD;
  const innerH = LOSS_H - 2 * LOSS_PAD;
  const gridYs = [0, 0.25, 0.5, 0.75, 1].map((t) => LOSS_PAD + t * innerH);

  const linePts = (values: number[]) => {
    if (values.length < 2) return '';
    return values
      .map((v, i) => {
        const x = LOSS_PAD + (i / (values.length - 1)) * innerW;
        const y = LOSS_PAD + (1 - (v - minL) / span) * innerH;
        return `${x},${y}`;
      })
      .join(' ');
  };

  return (
    <svg
      className="h-auto w-full max-w-[400px]"
      viewBox={`0 0 ${LOSS_W} ${LOSS_H}`}
      role="img"
      aria-label="Training and validation loss by epoch"
    >
      <rect width="100%" height="100%" rx={8} className="fill-[#0a0d12]" />
      <g className="stroke-white/10">
        {gridYs.map((y) => (
          <line key={y} x1={LOSS_PAD} x2={LOSS_W - LOSS_PAD} y1={y} y2={y} strokeWidth={1} />
        ))}
      </g>
      <polyline
        fill="none"
        className="stroke-cyan-400"
        strokeWidth={2}
        strokeLinecap="round"
        points={linePts(losses)}
      />
      {valLosses.length > 1 ? (
        <polyline
          fill="none"
          className="stroke-amber-400"
          strokeWidth={2}
          strokeLinecap="round"
          points={linePts(valLosses)}
        />
      ) : null}
    </svg>
  );
}

const SCAT_W = 280;
const SCAT_H = 220;
const SCAT_PAD = 32;

interface ScatterPlotProps {
  actual: number[];
  pred: number[];
  unit?: string;
  title?: string;
}

function ScatterPlot({ actual, pred, unit, title }: ScatterPlotProps) {
  if (actual.length === 0) return null;

  const all = [...actual, ...pred];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const inner = SCAT_W - 2 * SCAT_PAD;
  const innerH = SCAT_H - 2 * SCAT_PAD;

  const toXY = (a: number, p: number) => {
    const x = SCAT_PAD + ((p - lo) / span) * inner;
    const y = SCAT_PAD + (1 - (a - lo) / span) * innerH;
    return { x, y };
  };

  const xLabel = unit ? `predicted (${unit}) →` : 'predicted →';
  const yLabel = unit ? `↑ actual (${unit})` : '↑ actual';

  return (
    <svg
      className="h-auto w-full max-w-[280px]"
      viewBox={`0 0 ${SCAT_W} ${SCAT_H}`}
      role="img"
      aria-label={
        title
          ? `Actual versus predicted ${title} on hold-out set`
          : 'Actual versus predicted values on hold-out set'
      }
    >
      <rect width="100%" height="100%" rx={8} className="fill-[#0a0d12]" />
      <line
        x1={SCAT_PAD}
        y1={SCAT_H - SCAT_PAD}
        x2={SCAT_W - SCAT_PAD}
        y2={SCAT_PAD}
        className="stroke-white/20"
        strokeWidth={1}
        strokeDasharray="4 4"
      />
      <text
        className="fill-slate-500 font-mono text-[8px]"
        x={SCAT_W - SCAT_PAD}
        y={SCAT_H - 8}
        textAnchor="end"
      >
        {xLabel}
      </text>
      <text className="fill-slate-500 font-mono text-[8px]" x={SCAT_PAD} y={SCAT_PAD - 8}>
        {yLabel}
      </text>
      {actual.map((a, i) => {
        const { x, y } = toXY(a, pred[i]);
        return (
          <circle key={i} cx={x} cy={y} r={4} className="fill-sky-400/90" />
        );
      })}
    </svg>
  );
}

export { LossChart, ScatterPlot };
