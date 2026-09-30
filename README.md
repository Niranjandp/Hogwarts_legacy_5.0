# EVlove: Intelligent EV Route Charging Optimizer

![EVlove Dashboard Banner](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)

**EVlove** is an intelligent Electric Vehicle (EV) route charging optimization system designed to find the optimal charging strategy along a long-distance corridor. It minimizes travel time, waiting time, and charging cost by employing both Classical Dynamic Programming and Quantum-Inspired QUBO heuristics to find the optimal charging path.

## 🚀 Features

- **Multi-Objective Optimization**: Balances total journey time against total charging cost using customizable weights ($\alpha$ and $\beta$).
- **Dual Solver Architecture**: Races a classical Bellman Dynamic Programming engine against a Quantum-Inspired QUBO heuristic.
- **Dynamic Live Telemetry Simulation**: Simulates realistic real-time events such as station outages, queue wait times, and time-of-use peak pricing.
- **Interactive UI**: Built with Streamlit, providing real-time data visualization and route definition.
- **Multiple Routing Fallbacks**: Utilizes Google Maps Routes API with a seamless fallback to OSRM (free OpenStreetMap routing) and a static GPS waypoint backbone.
- **Detailed Analytics**: Offers interactive charts detailing the battery State-of-Charge (SoC) trajectory and time distribution for the journey.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/evolve-optimizer.git
   cd evolve-optimizer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables (Optional):**
   The application uses free OSRM by default, but to enable real-time traffic-aware Google Maps routing, create a `.env` file in the root directory:
   ```env
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
   OCM_API_KEY=your_openchargemap_api_key_here
   ```

## 🏃‍♂️ Running the Application

To start the interactive EVlove dashboard, run:
```bash
streamlit run dashboard/app.py
```

The application will launch on your default browser at `http://localhost:8501`.

## 🛠 Project Structure

- `dashboard/app.py`: The main Streamlit web application.
- `core/`: Contains the foundational models (`EV`, `Station`) and validation/objective functions.
- `data/`: Handles routing services (`route_engine`), station spatial placement (`station_finder`), and real-time noise simulation (`telemetry_sim`).
- `engines/`: Houses the core mathematical solvers (`classical_dp.py`, `quantum_qubo.py`).
- `experiments/`: Benchmark and parallel scaling execution suite (`race_runner.py`).
- `config.py`: Global constants, constants, API key loading, and system parameters.

## ☁️ Deployment

This project is fully ready for deployment on platforms like **Streamlit Community Cloud**, **Render**, or **Heroku**. 
Simply link your GitHub repository to the deployment platform. The included `requirements.txt` will automatically install the necessary packages.

## 📄 License

This project is open-sourced under the MIT License.
