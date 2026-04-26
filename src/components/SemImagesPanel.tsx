import { AnimatePresence, motion } from 'framer-motion';
import { useCallback, useRef, useState } from 'react';
import { Button } from './ui';

export interface SemImage {
  id: string;
  name: string;
  size: number;
  dataUrl: string;
  width: number;
  height: number;
  caption: string;
  uploadedAt: number;
}

interface SemImagesPanelProps {
  images: SemImage[];
  onAddFiles: (files: File[]) => void;
  onRemove: (id: string) => void;
  onCaptionChange: (id: string, caption: string) => void;
  onClearAll: () => void;
}

const MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function SemImagesPanel({
  images,
  onAddFiles,
  onRemove,
  onCaptionChange,
  onClearAll,
}: SemImagesPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [skipped, setSkipped] = useState<string[]>([]);

  const handleFiles = useCallback(
    (list: FileList | File[]) => {
      const arr = Array.from(list);
      const accepted: File[] = [];
      const rejected: string[] = [];
      for (const f of arr) {
        if (!f.type.startsWith('image/')) {
          rejected.push(`${f.name} — not an image`);
          continue;
        }
        if (f.size > MAX_FILE_SIZE_BYTES) {
          rejected.push(`${f.name} — over 12 MB`);
          continue;
        }
        accepted.push(f);
      }
      setSkipped(rejected);
      if (accepted.length > 0) onAddFiles(accepted);
    },
    [onAddFiles]
  );

  const onPickClick = () => inputRef.current?.click();

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) handleFiles(e.target.files);
    e.target.value = '';
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const onDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setDragActive(false);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer?.files?.length) handleFiles(e.dataTransfer.files);
  };

  return (
    <div className="flex flex-col gap-4">
      <p className="rounded-xl border border-amber-200/15 bg-amber-200/[0.04] px-3 py-2 text-[0.72rem] leading-relaxed text-amber-100/85">
        SEM images are stored in this browser session for reference and
        documentation. The current ANN consumes only scalar mix-design features
        — images are <em>not</em> fed into training.
      </p>

      <div
        role="button"
        tabIndex={0}
        onClick={onPickClick}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onPickClick();
          }
        }}
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={[
          'relative cursor-pointer rounded-2xl border-2 border-dashed px-4 py-8 text-center transition-colors duration-200',
          dragActive
            ? 'border-cyan-400/60 bg-cyan-400/[0.06]'
            : 'border-white/[0.12] bg-black/30 hover:border-white/[0.22] hover:bg-black/40',
        ].join(' ')}
        aria-label="Drop SEM images here or click to browse"
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="sr-only"
          onChange={onChange}
        />
        <div className="flex flex-col items-center gap-2">
          <span className="flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] font-mono text-xs uppercase tracking-[0.18em] text-slate-400">
            SEM
          </span>
          <p className="text-sm font-medium text-slate-200">
            Drop SEM images here or{' '}
            <span className="text-cyan-300 underline-offset-2 hover:underline">
              click to browse
            </span>
          </p>
          <p className="text-[0.68rem] text-slate-500">
            JPG / PNG / WEBP up to 12 MB each
          </p>
        </div>
      </div>

      <AnimatePresence>
        {skipped.length > 0 && (
          <motion.ul
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="flex flex-col gap-1 rounded-xl border border-red-400/20 bg-red-500/[0.06] px-3 py-2 font-mono text-[0.7rem] text-red-200/90"
          >
            {skipped.map((s, i) => (
              <li key={i}>· {s}</li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>

      {images.length > 0 ? (
        <>
          <div className="flex items-baseline justify-between gap-2">
            <p className="font-mono text-[0.7rem] text-slate-500">
              {images.length} image{images.length === 1 ? '' : 's'} ·{' '}
              {formatSize(images.reduce((s, i) => s + i.size, 0))}
            </p>
            <Button variant="ghost" size="sm" onClick={onClearAll}>
              Clear all
            </Button>
          </div>
          <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            <AnimatePresence initial={false}>
              {images.map((img) => (
                <motion.li
                  key={img.id}
                  layout
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.96 }}
                  transition={{ type: 'spring', stiffness: 360, damping: 28 }}
                  className="group relative flex flex-col overflow-hidden rounded-xl border border-white/[0.06] bg-black/40 transition-colors hover:border-white/[0.18]"
                >
                  <div className="relative aspect-square w-full overflow-hidden bg-black/60">
                    <img
                      src={img.dataUrl}
                      alt={img.caption || img.name}
                      className="h-full w-full object-cover"
                      loading="lazy"
                    />
                    <button
                      type="button"
                      onClick={() => onRemove(img.id)}
                      className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-md border border-white/15 bg-black/70 text-slate-300 opacity-0 transition-opacity hover:border-red-400/40 hover:text-red-200 focus:opacity-100 group-hover:opacity-100"
                      aria-label={`Remove ${img.name}`}
                    >
                      <span aria-hidden className="text-sm leading-none">
                        ×
                      </span>
                    </button>
                  </div>
                  <div className="flex flex-col gap-1 px-2.5 py-2">
                    <p
                      className="truncate text-[0.7rem] font-medium text-slate-200"
                      title={img.name}
                    >
                      {img.name}
                    </p>
                    <p className="font-mono text-[0.62rem] text-slate-500">
                      {img.width}×{img.height} · {formatSize(img.size)}
                    </p>
                    <input
                      type="text"
                      value={img.caption}
                      placeholder="Add caption…"
                      onChange={(e) => onCaptionChange(img.id, e.target.value)}
                      className="mt-1 rounded-md border border-white/[0.06] bg-black/40 px-2 py-1 text-[0.7rem] text-slate-200 placeholder:text-slate-600 focus:border-cyan-400/50 focus:outline-none"
                    />
                  </div>
                </motion.li>
              ))}
            </AnimatePresence>
          </ul>
        </>
      ) : (
        <p className="rounded-xl border border-white/[0.04] bg-white/[0.015] px-3 py-3 text-center text-[0.7rem] text-slate-500">
          No SEM images uploaded yet.
        </p>
      )}
    </div>
  );
}
