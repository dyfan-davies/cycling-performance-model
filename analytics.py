import streamlit as st
import streamlit.components.v1 as components

def inject_ga():
    """
    Injects GA4 code using a frontend component.
    """
    
    # We grab the tracking ID from the function or hardcode it
    GA_ID = "G-YXB3FZ1SLL"

    # This JavaScript:
    # 1. Loads the GA4 library.
    # 2. Configures it with your ID.
    # 3. CRITICAL: Tells GA to track the 'parent' window (your actual app),
    #    not the invisible iframe box holding this script.
    ga_js = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());

        // Configure GA4 to look at the parent window (the main app)
        gtag('config', '{GA_ID}', {{
            'page_location': window.parent.location.href,
            'page_title': window.parent.document.title
        }});
        
        console.log("GA4 Initialized for {GA_ID}");
    </script>
    """
    
    # Inject invisibly
    components.html(ga_js, height=0, width=0)
