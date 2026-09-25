/**
 * ThemeToggle.jsx
 * ================
 * Animated sun/moon toggle button for dark ↔ light theme switching.
 * Uses a smooth cross-fade (200ms) via the CSS transition on <html>.
 */
import { motion, AnimatePresence } from 'framer-motion';
import { Sun, Moon } from 'lucide-react';

export default function ThemeToggle({ isDark, onToggle }) {
  return (
    <button
      id="theme-toggle"
      onClick={onToggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="relative flex items-center justify-center w-8 h-8 rounded-lg border border-[var(--color-paper-border)] bg-[var(--color-paper-surface)] hover:bg-[var(--color-paper-elevated)] active:scale-95 transition-all"
    >
      <AnimatePresence mode="wait" initial={false}>
        {isDark ? (
          <motion.span
            key="sun"
            initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
            animate={{ rotate: 0, opacity: 1, scale: 1 }}
            exit={{ rotate: 90, opacity: 0, scale: 0.6 }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
            className="absolute"
          >
            <Sun className="w-4 h-4 text-[var(--color-status-warning)]" />
          </motion.span>
        ) : (
          <motion.span
            key="moon"
            initial={{ rotate: 90, opacity: 0, scale: 0.6 }}
            animate={{ rotate: 0, opacity: 1, scale: 1 }}
            exit={{ rotate: -90, opacity: 0, scale: 0.6 }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
            className="absolute"
          >
            <Moon className="w-4 h-4 text-[var(--color-sector-naval)]" />
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );
}
