#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 16:27:01 2025

@author: dyfanrhys
"""

import streamlit as st
import pathlib
import logging
import shutil

def inject_ga():
    """
    Injects the specific GA4 tag (G-YXB3FZ1SLL) into the Streamlit index.html.
    """
    # 1. Path to the Streamlit index.html file in your library installation
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    
    if not index_path.exists():
        logging.warning("Could not find Streamlit index.html. GA injection skipped.")
        return

    # 2. Read the existing HTML
    html_content = index_path.read_text()

    # 3. Stop if the tag is already present (prevents duplicates on re-runs)
    if "G-YXB3FZ1SLL" in html_content:
        return

    # 4. Your specific Google Analytics 4 Tag
    ga_tag = """
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-YXB3FZ1SLL"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', 'G-YXB3FZ1SLL');
    </script>
    """

    # 5. Insert the tag immediately before the closing </head> tag
    if "</head>" in html_content:
        new_html = html_content.replace("</head>", f"{ga_tag}\n</head>")
        
        try:
            # Write the modified HTML back to the file
            index_path.write_text(new_html)
            # Log success (visible in your terminal/console)
            print("GA4 tag (G-YXB3FZ1SLL) successfully injected into index.html")
        except PermissionError:
            # This handles environments where the filesystem is read-only
            logging.error("Permission denied: Could not modify index.html for GA.")