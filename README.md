# Cycling Performance Modeller

An advanced physics simulation engine designed to model cycling performance and optimise race strategies using Genetic Algorithms.

## 📝 Overview

This project achieves two primary objectives:
1.  **Performance Prediction:** Determines exactly how long a specific course takes to complete based on a rider's power profile using a custom physics engine.
2.  **Strategy Optimisation:** Identifies the fastest possible pacing strategy for a given route using a Genetic Algorithm, without increasing the rider's average power input.

**Live Demo:** http://cycling-performance-model.streamlit.app

## ⚙️ Key Features

* **⏱️ Ride Time Calculator:** Predicts finish time given a fixed power output, accounting for mass, aerodynamics, and rolling resistance.
* **⚡ Power Estimator:** Reverse-calculates the required average power (Watts) to achieve a specific target finish time using bisection search.
* **📈 Race Strategy Optimiser:** Uses an evolutionary algorithm to distribute energy efficiently across a course, saving time compared to a constant-power effort.

## 🛠️ System Architecture

The application is built on a modular, object-oriented design:

* **Physics Engine (`physics_engine.py`):** A vectorised, time-stepping solver that integrates mechanical forces (Gravity, Drag, Rolling Resistance) and physiological states (Skiba W' Model) at 1-second intervals. It calculates dynamic variables such as **Apparent Wind Yaw** and **Cornering Limits** rather than relying on static averages.
* **Optimiser (`optimizer.py`):** Implements a Genetic Algorithm with selection, crossover, mutation, and elitism to evolve optimal pacing strategies.
* **GPX Tool (`gpx_tool.py`):** An ETL pipeline that cleans raw GPS data using Haversine distance calculations and Savitzky–Golay signal smoothing to remove GPS jitter.
* **Frontend:** Built with Streamlit and PyDeck for interactive 3D data visualization.

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dyfan-davies/cycling-performance-model.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run Home.py
    ```

## 🧠 The Physics Model

The engine balances four primary mechanical forces at every time step ($dt=1s$):

1.  **Gravity:** $F_{grav} = m \cdot g \cdot \sin(\theta)$
2.  **Rolling Resistance:** $F_{roll} = C_{rr} \cdot m \cdot g \cdot \cos(\theta)$
3.  **Aerodynamic Drag:** $F_{drag} = 0.5 \cdot \rho \cdot C_d A \cdot v^2$
4.  **Propulsion:** $F_{prop} = (P \cdot \eta) / v$

### Advanced Logic
* **Cornering Physics:** The model scans the GPX path ahead to calculate the radius of curvature ($r$) and enforces speed limits ($v_{max} = \sqrt{\mu g r}$) to prevent unrealistic cornering speeds.
* **Physiology Engine:** Implements the **Skiba $W'$ Model** to track anaerobic work capacity. It allows the rider to surge above Critical Power on climbs but forces recovery or "bonking" if the energy battery depletes.

## 🧬 Genetic Algorithm Methodology

To solve the pacing problem, the system evolves a population of "riders" over multiple generations:
1.  **Population:** Randomly initialized power strategies (genes).
2.  **Selection:** Strategies are simulated; fastest times are selected as parents.
3.  **Crossover:** Parent strategies are spliced to create new "child" strategies.
4.  **Mutation:** Random variations are introduced to avoid local optima.
5.  **Elitism:** The best strategies are preserved across generations.

## 📄 License

This project is licensed under the [GNU General Public License v3.0] - see the LICENSE file for details.

---

### Author
**Dyfan Davies** *MPhys Physics (Cardiff University) | MSc Sports Engineering (Sheffield Hallam University)*
[Connect on LinkedIn](https://www.linkedin.com/in/dyfan-davies/)
