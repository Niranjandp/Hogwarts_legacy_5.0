# 🚀 EVolve Deployment Guide (Desktop & Mobile Ready)

EVolve is an Intelligent EV Route & Charging Optimizer powered by Classical Dynamic Programming and Quantum QUBO simulated annealing. It is configured to run on both desktop and mobile web browsers with responsive viewports, touch-friendly UI controls, and live telemetry synchronization.

---

## 📱 Immediate Mobile & Desktop Access (Local Network)

If your mobile phone or tablet is connected to the same local Wi-Fi / LAN network as your computer, you can access the dashboard immediately without any cloud setup:

- **Desktop (Local):** [`http://localhost:8506`](http://localhost:8506)
- **Mobile (Same Wi-Fi):** `http://172.16.27.51:8506`

*(Tip: You can bookmark the URL on your mobile browser's home screen for an app-like fullscreen experience).*

---

## 🌐 Option 1: Streamlit Community Cloud (Recommended - Free & Permanent HTTPS)

Streamlit Community Cloud provides free, high-performance hosting with zero server maintenance.

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Configure production deployment and mobile responsiveness"
   git push origin main
   ```
2. **Open [share.streamlit.io](https://share.streamlit.io):**
   - Sign in with your GitHub account.
   - Click **"New app"**.
   - Select your repository: `Niranjandp/Hogwarts_legacy_5.0`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
   - Advanced settings (Optional): Add your Google Maps API key or OCM API key under Secrets if needed:
     ```toml
     GOOGLE_MAPS_API_KEY = "your-api-key"
     OCM_API_KEY = "your-ocm-key"
     ```
3. **Click "Deploy":**
   - Streamlit Cloud will install dependencies from `requirements.txt` (including `dwave-neal`, `streamlit-folium`, `plotly`, `pandas`, `numpy`).
   - Your public URL will be live at `https://<your-app-name>.streamlit.app`, accessible from any smartphone, tablet, or desktop across the globe.

---

## 🐳 Option 2: Docker Container Deployment

The repository includes a production-ready `Dockerfile`:

1. **Build the container image:**
   ```bash
   docker build -t evolve-ev-optimizer:latest .
   ```
2. **Run the container:**
   ```bash
   docker run -d -p 8506:8506 --name evolve-app evolve-ev-optimizer:latest
   ```
3. Access at `http://localhost:8506` or `http://<server-ip>:8506`.

---

## ☁️ Option 3: Render / Railway / Cloud Run

The repository includes a `Procfile` and `.streamlit/config.toml`:

### Render
1. Create a **New Web Service** linked to your GitHub repo.
2. Select **Python 3** environment.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0`

### Railway
1. Click **"New Project"** ➔ **"Deploy from GitHub repo"**.
2. Railway detects the `Dockerfile` or `Procfile` automatically and provides a public `.up.railway.app` HTTPS domain.

---

## ⚡ Instant Public Mobile Tunnel (For Demos & Testing)

To share a live temporary link directly from your local machine to any phone on cellular data (4G/5G):

```bash
cmd /c "npx -y localtunnel --port 8506"
```
Or with ngrok:
```bash
ngrok http 8506
```

---

## 🎨 Mobile-First & Desktop Optimizations Included

- **Dynamic Viewport:** Folium interactive route map automatically uses 100% container width (`use_container_width=True`) rather than fixed desktop pixels.
- **Auto-Collapsing Sidebar:** On mobile viewports (<768px), the sidebar auto-collapses into the hamburger menu to avoid blocking the route view.
- **Horizontal Scroll Protection:** Dataframes and charging sequence tables feature touch-friendly horizontal momentum scrolling.
- **Touch-Optimized Targets:** Buttons, selectors, and preset toggles meet WCAG minimum 44px tap targets.
- **Dark Glassmorphism:** Optimized battery-efficient dark OLED theme (`#0b0f19`) with blurred translucent cards.
