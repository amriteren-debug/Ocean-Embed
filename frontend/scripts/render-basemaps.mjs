/**
 * render-basemaps.mjs
 * Renders 6 Plotly geo base map PNGs using realistic satellite-style colors.
 * Output: src/assets/satellite-1.png ... satellite-6.png (1920x1080 each)
 */
import puppeteer from 'puppeteer';
import { mkdirSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, '..', 'src', 'assets');
mkdirSync(OUT_DIR, { recursive: true });

// Realistic satellite palette — dark navy ocean, earthy terrain, subtle borders
const LAND_COLOR    = '#3d4a2e';   // dark olive-green (vegetation/terrain from orbit)
const COAST_COLOR   = 'rgba(120, 140, 100, 0.6)';
const OCEAN_COLOR   = '#0a1628';   // near-black deep ocean, like real satellite
const LAKE_COLOR    = '#0d1f3c';
const BG_COLOR      = '#040d1a';   // space-black background

const VARIANTS = [
  { lon: [45, 100], lat: [-10, 30],  label: 'Full domain' },
  { lon: [45, 80],  lat: [-5, 30],   label: 'Arabian Sea focus' },
  { lon: [65, 100], lat: [-10, 25],  label: 'Bay of Bengal focus' },
  { lon: [48, 98],  lat: [-10, 15],  label: 'Southern strip' },
  { lon: [50, 100], lat: [5, 30],    label: 'Northern strip' },
  { lon: [55, 95],  lat: [-5, 25],   label: 'Central crop' },
];

function buildHtml(lonRange, latRange) {
  const layout = {
    autosize: true,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: BG_COLOR,
    plot_bgcolor: BG_COLOR,
    geo: {
      projection: { type: 'equirectangular' },
      lonaxis: { range: lonRange },
      lataxis: { range: latRange },
      showcoastlines: true,
      coastlinecolor: COAST_COLOR,
      coastlinewidth: 1.5,
      showland: true,
      landcolor: LAND_COLOR,
      showocean: true,
      oceancolor: OCEAN_COLOR,
      showlakes: true,
      lakecolor: OCEAN_COLOR,
      bgcolor: BG_COLOR,
      showframe: false,
      resolution: 50,
      showsubunits: false,
      showcountries: false,
    },
  };

  const data = [{
    type: 'scattergeo',
    lat: [null],
    lon: [null],
    showlegend: false,
    hoverinfo: 'none',
    marker: { opacity: 0 },
  }];

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>*{margin:0;padding:0;box-sizing:border-box}html,body{width:1920px;height:1080px;overflow:hidden;background:${BG_COLOR}}#plot{width:1920px;height:1080px}</style>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\/script>
</head><body>
<div id="plot"></div>
<script>
Plotly.newPlot('plot',${JSON.stringify(data)},${JSON.stringify(layout)},{displayModeBar:false,responsive:false,scrollZoom:false})
  .then(()=>{window.__plotReady=true;});
<\/script>
</body></html>`;
}

async function renderAll() {
  console.log('Launching Chromium...');
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox','--disable-setuid-sandbox','--disable-gpu'],
  });

  for (let i = 0; i < VARIANTS.length; i++) {
    const v = VARIANTS[i];
    console.log(`Rendering satellite-${i+1}.png  [${v.label}]...`);
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.setContent(buildHtml(v.lon, v.lat), { waitUntil: 'networkidle0' });
    await page.waitForFunction('window.__plotReady === true', { timeout: 30000 });
    await new Promise(r => setTimeout(r, 800));
    const outPath = path.join(OUT_DIR, `satellite-${i+1}.png`);
    await page.screenshot({ path: outPath, type: 'png', fullPage: false });
    console.log(`  Saved: ${outPath}`);
    await page.close();
  }

  await browser.close();
  console.log('\nDone. All 6 base map images saved to src/assets/');
}

renderAll().catch(err => { console.error(err); process.exit(1); });
