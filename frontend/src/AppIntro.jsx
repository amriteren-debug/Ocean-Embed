import { useState, useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import LoadingScreen from './components/LoadingScreen';

// ── Vite static asset imports (only the picked one is actually fetched at runtime) ──
import sat1 from './assets/satellite-1.png';
import sat2 from './assets/satellite-2.png';
import sat3 from './assets/satellite-3.png';
import sat4 from './assets/satellite-4.png';
import sat5 from './assets/satellite-5.png';
import sat6 from './assets/satellite-6.png';

const SATELLITE_URLS = [sat1, sat2, sat3, sat4, sat5, sat6];

// Pick one at random, once per page load — locked for the session
const PICKED_URL = SATELLITE_URLS[Math.floor(Math.random() * SATELLITE_URLS.length)];
const DISPLAY_DURATION_MS = 5000;

/**
 * State machine:
 *   "loading-initial"  → image onLoad fires  → "image"
 *   "image"            → 5s setTimeout        → "loading-final"
 *   "loading-final"    → 500ms setTimeout     → "app"
 */
export default function AppIntro({ children }) {
  const [phase, setPhase] = useState('loading-initial');
  const imgRef = useRef(null);

  // Preload only the one selected image.
  // If it loads before we even mount (cached), onLoad won't fire — check readiness immediately.
  useEffect(() => {
    const img = new Image();
    img.src = PICKED_URL;

    const onLoad = () => setPhase('image');
    const onError = () => {
      // If image fails (e.g., not generated yet), skip straight to app
      console.warn('[AppIntro] Satellite image failed to load, skipping splash.');
      setPhase('app');
    };

    if (img.complete && img.naturalWidth > 0) {
      // Already cached
      setPhase('image');
    } else {
      img.addEventListener('load', onLoad);
      img.addEventListener('error', onError);
    }

    imgRef.current = img;
    return () => {
      img.removeEventListener('load', onLoad);
      img.removeEventListener('error', onError);
    };
  }, []);

  // When entering "image" phase, start the 5-second display timer
  useEffect(() => {
    if (phase !== 'image') return;
    const t = setTimeout(() => setPhase('loading-final'), DISPLAY_DURATION_MS);
    return () => clearTimeout(t);
  }, [phase]);

  // Short delay on "loading-final" before revealing the app
  useEffect(() => {
    if (phase !== 'loading-final') return;
    const t = setTimeout(() => setPhase('app'), 1800);
    return () => clearTimeout(t);
  }, [phase]);

  // Once app phase reached, render children directly (no wrapper overhead)
  if (phase === 'app') return children;

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#0a0f18', overflow: 'hidden' }}>
      <AnimatePresence mode="wait">

        {/* ── Loading screen (initial & final) ── */}
        {(phase === 'loading-initial' || phase === 'loading-final') && (
          <motion.div
            key="loading"
            className="absolute inset-0"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Wrap in relative container so LoadingScreen's absolute positioning works */}
            <div style={{ position: 'relative', width: '100%', height: '100%' }}>
              <LoadingScreen />
            </div>
          </motion.div>
        )}

        {/* ── Raw satellite image, full-screen, no overlay ── */}
        {phase === 'image' && (
          <motion.div
            key="satellite"
            className="absolute inset-0"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            style={{ background: '#000' }}
          >
            <img
              src={PICKED_URL}
              alt=""
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                display: 'block',
              }}
            />
          </motion.div>
        )}

      </AnimatePresence>
    </div>
  );
}
