# Copper slag concrete — compressive strength ANN.

Browser-based app that trains a small neural network (**TensorFlow.js**) to predict **compressive strength** from mix design and SEM-style scalars: cement, water, copper slag %, curing days, porosity %, and crack density. All training and inference run **client-side** (no server).

**Repository:** [github.com/allwinromario/Copper-slag-concrete](https://github.com/allwinromario/Copper-slag-concrete)

---

## Prerequisites

1. **[Node.js](https://nodejs.org/)** **18.x or newer** (20 LTS recommended). Check your version:

   ```bash
   node -v
   ```

2. **npm** (bundled with Node.js):

   ```bash
   npm -v
   ```

3. **Git** (to clone and push). Optional if you already have the project folder.

---

## Step-by-step setup

### 1. Get the code

**Option A — Clone (recommended)**

```bash
git clone https://github.com/allwinromario/Copper-slag-concrete.git
cd Copper-slag-concrete
```

**Option B — You already have the folder**

```bash
cd /path/to/Copper-slag-concrete
```

### 2. Install dependencies

From the project root (where `package.json` is):

```bash
npm install
```

This installs React, Vite, TensorFlow.js, Tailwind CSS, Framer Motion, and TypeScript tooling.

### 3. Run the app in development

```bash
npm run dev
```

- Vite prints a local URL (usually **http://localhost:5173**).
- Open that URL in **Chrome**, **Edge**, or **Firefox** (WebGL / WASM are used by TensorFlow.js).

To stop the server, press `Ctrl+C` in the terminal.

### 4. Production build (optional)

Create an optimized build in the `dist/` folder:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

---

## Using the app (short flow)

1. **Training data** — Generate sample CSV data or load the demo; use **Training setup** (sidebar) for epochs, learning rate, and validation fraction.
2. **Save inputs** — Set the six inference fields, then click **Save inputs** so **Train ANN** appears (narrow layout).
3. **Train** — Run training; **Diagnostics** show loss and hold-out metrics.
4. **Predict** — Use **Generate predicted strength**; the **compressive strength** readout updates below the workspace.

Inference values are stored in **localStorage** in your browser so they survive refresh (see app behavior in the UI).

---

## Troubleshooting

| Issue | What to try |
|--------|----------------|
| `npm install` errors | Use Node 18+; delete `node_modules` and `package-lock.json`, then `npm install` again. |
| Blank / slow first train | First load downloads TF.js WASM; wait a few seconds; keep the tab focused. |
| Port 5173 in use | Vite will suggest another port, or run `npm run dev -- --port 3000`. |

---

## Tech stack

- **React 18** + **TypeScript**
- **Vite 5**
- **TensorFlow.js** (in-browser training)
- **Tailwind CSS v4** + **Framer Motion**

---

## License

This project is provided as-is for research and education. Validate any predictions with proper laboratory testing before real-world use.
