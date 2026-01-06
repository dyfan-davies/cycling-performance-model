import streamlit as st
import analytics

analytics.inject_ga()

st.set_page_config(
    page_title="Cycling Performance Modeller",
    page_icon="🚴",
    layout="centered"
)

st.title("Cycling Performance Modeller")

# --- INTRO ---
st.markdown("""
**Analyse your cycling performance on any route.** Simply upload a **GPX file** (exported from Strava, Garmin, RideWithGPS, etc.) or test the tool immediately using one of the built-in demo files.

This model uses real-world physics, accounting for your power, weight, gravity and wind to predict exactly how you will perform on race day.

### Which tool should I use?

* **⏱️ Ride Time Calculator**:
    Input your power to see how fast you will ride.
    Best for answering: *"If I hold 200 Watts, what time will I finish?"*

* **⚡ Power Plan Estimator**:
    Input your target time to see the required power.
    Best for answering: *"What average wattage do I need to ride this course in 1 hour?"*

* **📈 Race Strategy Optimiser**:
    Uses a Genetic Algorithm to generate an optimised pacing plan.
    Instead of riding at a constant speed, this tool calculates exactly where to push hard (hills) and where to recover to minimize your total time.

*Select a tool below to begin!*
\n

""")

#st.markdown("---")

# --- DISCLAIMER SECTION ---
with st.expander("ℹ️ Important: Read about the 'Constant Power' assumption"):
    st.markdown("""
    **1. The Assumption**
    The *Ride Time Calculator* and *Power Estimator* tools assume a **Constant Power Output**. They calculate what happens if you hold the exact same wattage for every second of the ride—never surging, never resting.

    **2. The Reality**
    Real riding is dynamic. If you generally average **100 Watts** but hit a steep gradient, you would naturally surge to **150 Watts+** to clear the obstacle.
    
    **3. The Consequence**
    Because the basic tools strictly apply your average power against gravity, they may predict a **"Stall"** (failure to finish) on steep hills if your input power is too low, even if you could physically ride them by surging.
    
    👉 **Recommendation:** For a realistic plan that accounts for surging on climbs and recovering on descents, use the **Race Strategy Optimiser**.
    """)

st.markdown("---")

# --- TOOL 1: RIDE TIME CALCULATOR ---
st.header("⏱️ Simulate Ride Time")
st.markdown("**Predict your finish time based on your power.**")
st.info("""
Simply enter your power output (Watts). The physics engine will calculate how long the course will take you.
""")
if st.button("Go to Time Calculator"):
    st.switch_page("pages/1_Ride_Time_Calculator.py")

st.markdown("---")

# --- TOOL 2: POWER ESTIMATOR ---
st.header("⚡ Calculate Average Power")
st.markdown("**Calculate the watts needed for a specific time.**")
st.info("""
Enter a finish time to reverse-calculate the required power. You can use this to plan for a future goal, or enter a result from a past ride to estimate the average power you produced on this course.""")
if st.button("Go to Power Estimator"):
    st.switch_page("pages/2_Power_Estimator.py")

st.markdown("---")

# --- TOOL 3: RACE OPTIMISER ---
st.header("📈 Race Strategy Optimiser")
st.markdown("**Generate a scientifically optimal pacing plan.**")
st.info("""
Upload a GPX file and let the Genetic Algorithm determine the most efficient way to distribute your energy across the course. 

Curious how the Genetic Algorithm works? See **About the Project**
""")
if st.button("Start Strategy Optimiser"):
    st.switch_page("pages/3_Race_Optimiser.py")
    
    
    # --- SIDEBAR FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #666666; font-size: 12px;">
        Designed & Engineered by<br>
        <strong>Dyfan Davies</strong>
    </div>
    """,
    unsafe_allow_html=True
)