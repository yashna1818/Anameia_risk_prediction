# Frontend Architecture: Tools & Techniques

The Hematology AI dashboard client interface is built entirely using vanilla web technologies, omitting heavy frameworks to maximize performance, load speeds, and maintainability.

---

## 1. Core Stack & Libraries
* **HyperText Markup Language (HTML5):** Used for semantic structure (`<aside>`, `<main>`, `<section>`, `<header>`, etc.), ensuring proper accessibility (A11y) and SEO parsing.
* **Cascading Style Sheets (CSS3):** Powers the entire visual presentation layer and responsive grid layouts without Tailwind or utility framework overhead.
* **Vanilla JavaScript (ES6+):** Manages reactivity, asynchronous API communications, state storage, and dynamic DOM rendering.
* **Google Fonts API:** Imports the modern sans-serif typography families:
  * **Outfit:** Used for large display headers and branding elements (futuristic/geometric geometry).
  * **Inter:** Used for body text, numbers, and data tables (highly readable screen-optimized typeface).
* **FontAwesome CDN (v6.4.0):** Provides scalable vector icons to represent medical, biological, and technical UI actions.

---

## 2. CSS Design Systems & Techniques

```
  🎨 Dark Mode Glassmorphic Design System
  ├─ Custom Properties (:root variables for colors, blur values, grids)
  ├─ Flexbox & Grid (100vh responsive dual-pane layout)
  ├─ Glassmorphism (backdrop-filter: blur, translucent borders)
  └─ Micro-animations (CSS keyframe pulses, button hover scaling)
```

### A. Custom CSS Variable Tokens
CSS variables are defined at the `:root` level of `style.css` to keep a strict, premium design standard:
* **Backgrounds:** `--bg-app` (`#0b0f19`), `--bg-card` (`#151c2d`), `--border-color` (`rgba(255, 255, 255, 0.08)`).
* **Typography:** `--font-header` (`'Outfit'`), `--font-body` (`'Inter'`).
* **Visual States:** Translucent colors for glow effects and indicators (e.g., active tabs, status pulses).

### B. Glassmorphism & UI Layers
* **Translucent Cards:** Cards use a semi-transparent dark background coupled with a fine, light border (`1px solid rgba(255,255,255,0.06)`).
* **Frosted Blurs:** CSS `backdrop-filter: blur(12px)` creates a depth effect when overlapping items, typical of modern luxury dashboard interfaces.
* **Gradient Accents:** Subtle linear-gradients are applied to the brand logo, primary buttons, and target risk cards.

### C. Layout Grids & Responsiveness
* **Dual-Pane Sidebar Layout:** A fixed `aside` combined with a scrolling `main` content pane utilizing a CSS flexbox wrapper to fill `100vh`.
* **Grid Grids (`.grid-2col`, `.explorer-grid`):** Grid properties adapt to different viewports using media-queries:
  ```css
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
  ```

### D. Animations & Transitions
* **Pulsing Status Indicator (`.pulse-dot`):** Uses CSS `@keyframes` scaling and opacity changes to visually represent connection status.
* **Interactive Hover Scaling:** Standardizes buttons, nav items, and genes cards with `transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1)` to provide instant micro-feedback on cursor interaction.

---

## 3. JavaScript Reactive Logic & API Techniques

### A. Single Page Application (SPA) Routing
* Tabs are switched client-side using `data-tab` attributes on sidebar selectors.
* Navigation event listeners add/remove the `.active` CSS utility class, instantly toggling visibility between dashboard panels without page reloads.

### B. Asynchronous Communications (Fetch API)
* Uses the native `async/await` Fetch API to interact with backend endpoints:
  * `GET /api/preloaded-patients` — Fetches pre-configured medical models.
  * `POST /api/predict` — Sends patient input arrays and returns predictions and genomic meal plans.
  * `POST /api/train` — Retrains the pipeline models on the NHANES/GWAS dataset and returns fresh scores.
  * `POST /api/parse-vcf` — Parses uploaded genomic files.
  * `POST /api/nutrition-recommend` — Provides macro-oriented nutrition advice.

### C. VCF File Stream Reading
* Enlists a `FileReader` object to read local `.vcf` files in the browser.
* Extracts chromosome positions, reference alleles, and genotype variables before sending a lightweight JSON payload to the server, protecting memory bounds.

### D. Real-Time Form Calculation
* Integrates input listeners on height (`BMXHT`) and weight (`BMXWT`) selectors.
* Automatically computes and displays the Body Mass Index (BMI):
  $$\text{BMI} = \frac{\text{Weight (kg)}}{\left(\frac{\text{Height (cm)}}{100}\right)^2}$$
  This updates UI states in real time before the patient submits the form.
