/**
 * useTheme.js
 * ===========
 * Provides theme state (dark / light) persisted to localStorage.
 * Dark is the default on first load.
 * Applies 'data-theme' attribute to <html> so CSS variables cascade.
 */
import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'ocean-predictor-theme';
const DEFAULT_THEME = 'dark';

export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) ?? DEFAULT_THEME;
    } catch {
      return DEFAULT_THEME;
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {}
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(t => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  return { theme, toggleTheme, isDark: theme === 'dark' };
}
