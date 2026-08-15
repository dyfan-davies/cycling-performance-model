#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import analytics

# Inject Google Analytics (Optional tracking code)
def inject_ga():
    # Replace G-XXXXXXXXXX with your actual Tracking ID
    GA_ID = "G-XXXXXXXXXX"
    
    # Note: Streamlit runs components in an iframe, so this isn't perfect, 
    # but it works without crashing the server.
    ga_code = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{GA_ID}');
    </script>
    """
    st.components.v1.html(ga_code, height=0, width=0)

# --- PAGE CONFIGURATION ---
# Sets the browser tab title and icon.
st.set_page_config(page_title="About the Engineer", page_icon="🚴")

# --- MAIN TITLE SECTION ---
st.title("👨‍💻 About the Engineer")

# Bio Content
st.markdown("""
### Designed & Engineered by Dyfan Davies

Hey! I’m Dyfan, thanks for looking at my project! 

I grew up playing sport and was always curious about why things worked the way they did. That curiosity pulled me from the pitch into the lab, leading me to study physics and eventually sports engineering. I’m into all kinds of projects, not just sport. I really enjoy taking a tough question or a pile of data and turning it into something real. Whether that’s a tool you can actually play around with or just a clear graphic, I enjoy the process of making sense of the numbers and putting them in simple terms.

I built this application as a side project to explore performance questions through modelling and analysis. Through it, I explore questions around modelling, optimisation and performance analysis, from simulating dynamic systems to testing pacing strategies using genetic algorithms.

Thanks again for taking the time to look at my project! I have some more info on my background below. 

Please reach out if you have any questions or would like to discuss anything; my details are at the bottom of the page!
### 💼 My Background
I hold an MPhys in Physics from Cardiff University and an MSc in Sports Engineering from Sheffield Hallam University. 
My career has given me the opportunity to work across projects in both academia and industry. I've worked on projects ranging from fundamental research in controlled laboratory conditions to the implementation of critical technology in messy, real-world sporting environments. This work has taken me to stadiums and facilities across 15 countries spanning 3 continents.

**Key Highlights:**
* My Master's research on youth football head impacts became the first published paper from a FIFA partnership.
* Awarded the Centre for Sports Engineering Research Prize.
* Served as a member of the New Zealand Rugby Research Advisory Panel.
""")

# --- RESEARCH PROJECTS SECTION (EXPANDERS) ---
# Using st.expander allows the user to click to reveal details, keeping the page clean.
st.subheader("🔍 Research & Technical Projects")
st.markdown("A selection of previous projects that I've been involved in:")

with st.expander("🏉 Concussion Management Pathway (New Zealand Rugby)", expanded=True):
    st.markdown("""
    **Project:** *Implementing and Coordinating the CMP across Four Provincial Unions*
    
    I coordinated and supported the implementation and evaluation of the Concussion Management Pathway across four Provincial Unions. Leading a cross-disciplinary team, I managed the collection, processing, and cleaning of suspected Traumatic Brain Injury incident data.
    
    My role involved performing statistical analyses to generate insights for decision-making and directing the development of a custom app to streamline data entry. I also led the integration of independent player databases into a centralised tracking system to improve operational workflows.
    """)

with st.expander("⚽ Head Impacts in Youth Football (FIFA & Sheffield Hallam University)"):
    st.markdown("""
    **Project:** *Effect of Football Size and Mass on Head Accelerations*
    
    I investigated how football size and mass affect head accelerations during headers in youth players. I conducted field research, developed a ball trajectory model, and used a crash test dummy for experimental trials.
    
    My findings directly informed FIFA on ball size recommendations for youth football. The work was presented at the ISEA Conference and published in the conference proceedings [(DOI: 10.3390/proceedings2020049029)](https://doi.org/10.3390/proceedings2020049029).
    """)

with st.expander("🏉 Rugby Head Impact Study (World Rugby & New Zealand Rugby)"):
    st.markdown("""
    **Project:** *Women's Community Rugby Head Impact Study*
    
    I led a team collecting head impact data in community rugby using instrumented mouthguards. We captured real-time player acceleration during competitive games, helping build the largest global database of comparable rugby impact data.
    """)

with st.expander("📏 Laser Linewidth Measurement System (Cardiff University)"):
    st.markdown("""
    **Project:** *Precision Instrumentation & Optics*
    
    For my final-year project, I designed, fabricated, and tested a linewidth measurement system for VCSELs (Vertical-Cavity Surface-Emitting Lasers). VCSEL linewidth value is a key factor for accuracy in miniature atomic clocks, commonly used in small or power-constrained satellite applications.
    """)

with st.expander("🏏 Physics of Sport: Bat-Ball Collisions (Cardiff University)"):
    st.markdown("""
    **Project:** *High-Speed Impact Analysis*
    
    I analysed cricket bat-ball collision dynamics using a high-speed 2D camera. I investigated how pivot point movement and the centre of percussion affect bat performance and ball response.
    """)

with st.expander("🛠️ Sports Equipment Design & Simulation"):
    st.markdown("""
    **Project:** *UCI-Compliant Bike & Custom Protection*
    
    I designed and simulated a UCI-compliant racing bicycle and a custom-fit field hockey shin-pad with an adjustable airbag. I used SolidWorks, ANSYS (CFD/FEA) tools, and physical testing to simulate and validate performance.
    """)

st.markdown("---")

# --- COLLABORATIONS SECTION (COLUMNS) ---
# Using 3 columns to organize the lists of organizations horizontally.
st.subheader("🤝 Collaborations")
st.markdown("I've been fortunate enough to work on projects with a range of leading sporting organisations, technology companies and research institutions:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### **Sporting Organisations & Governing Bodies**")
    st.markdown("""
    * **ACC** (*NZ national injury and rehabilitation agency*)
    * **Great Britain Cycling Team (British Cycling)**
    * **FIFA**
    * **New Zealand Rugby**
    * **NZ Provincial Rugby Unions** (*North Harbour, Hawke's Bay, Otago & Wairarapa-Bush*)
    * **Team England**
    * **UEFA**
    * **UKSI**
    * **World Rugby**
    """)

with col2:
    st.markdown("#### **Sports Technology & Industry**")
    st.markdown("""
    * **Hawk-Eye Innovations**
    * **HITIQ**
    * **Labosport**
    * **Prevent Biometrics**
    """)

with col3:
    st.markdown("#### **Academic & Research Institutions**")
    st.markdown("""
    * **Auckland University of Technology**
    * **Cardiff University**
    * **Sheffield Hallam University**
    * **University of Otago**
    """)

st.markdown("---")

# --- CONTACT SECTION ---
st.markdown("""
### 📬 Contact me

**Want a deeper dive?** The web demo has limits to ensure server stability, but I can run high-fidelity simulations offline. Reach out if you want a run with ***custom manual course segmentation*** or ***expanded generations and population values***.

**Feedback & Bugs:** Your feedback helps improve this tool. If you spot a discrepancy in the physics, find a bug, or have a feature request or suggestion, please let me know!

**Work & Chat:** I’m always happy to connect with people working in ***Engineering***, ***Research*** and ***Research & Development***. If you’d like to discuss science, sport, engineering or anything in between, feel free to connect with me on LinkedIn below!

[**Connect on LinkedIn**](https://www.linkedin.com/in/dyfan-davies/)

""")

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
