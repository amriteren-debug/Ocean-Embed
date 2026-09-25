import { motion } from 'framer-motion';
import { Compass } from 'lucide-react';

export default function Header() {
  return (
    <header className="flex items-center gap-3">
      <motion.div
        className="flex items-center justify-center w-9 h-9 rounded-lg shadow-sm"
        style={{ background: 'var(--color-accent)' }}
        animate={{ rotate: [0, 2, -2, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
      >
        <Compass className="w-5 h-5 text-white" />
      </motion.div>
      <div className="flex flex-col justify-center">
        <h1
          className="text-lg font-bold tracking-tight font-display leading-none"
          style={{ color: 'var(--color-ink-bright)' }}
        >
          Ocean Predictor
        </h1>
        <span
          className="text-[9px] font-bold tracking-[0.15em] uppercase leading-none mt-1"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          Subsurface Thermal Analysis
        </span>
      </div>
    </header>
  );
}
