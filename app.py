import streamlit as st
import cv2
import numpy as np
import json
import os
import time
from scanner_module import ShapeScanner

st.set_page_config(layout="wide", page_title="ระบบเช็คลิสต์เครื่องมือ")

# --- CSS ตกแต่งตาม Theme Palette ---
# Palette: #FFFFFF, #E5E5E5, #C7C7C7, #525252, #2C2C2C, #E81D23
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #FFFFFF;
        color: #2C2C2C;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #2C2C2C !important;
        font-weight: 700 !important;
    }

    /* Dividers */
    hr {
        border: none !important;
        border-top: 1px solid #C7C7C7 !important;
        margin: 1rem 0 !important;
    }

    /* Primary Buttons (#E81D23) */
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background-color: #E81D23 !important;
        color: #FFFFFF !important;
        border: 1px solid #E81D23 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 5px rgba(232, 29, 35, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: #c7161b !important;
        border-color: #c7161b !important;
        box-shadow: 0 4px 10px rgba(232, 29, 35, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Secondary / Default Buttons */
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {
        background-color: #E5E5E5 !important;
        color: #2C2C2C !important;
        border: 1px solid #C7C7C7 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #C7C7C7 !important;
        border-color: #525252 !important;
        color: #2C2C2C !important;
    }

    /* Radio Group Styling */
    div[data-testid="stRadio"] > div {
        background-color: #E5E5E5;
        border: 1px solid #C7C7C7;
        border-radius: 8px;
        padding: 6px 14px;
        gap: 16px;
    }
    div[data-testid="stRadio"] label {
        color: #2C2C2C !important;
        font-weight: 500;
    }

    /* Checkbox & Selection */
    div[data-testid="stCheckbox"] {
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 10px;
    }

    /* Card Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #C7C7C7 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #525252 !important;
        box-shadow: 0 2px 8px rgba(44, 44, 44, 0.08) !important;
    }

    /* Badges & Text */
    .badge-category {
        background-color: #E5E5E5;
        color: #2C2C2C;
        border: 1px solid #C7C7C7;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 500;
        display: inline-block;
    }
    .badge-score {
        color: #E81D23;
        font-weight: 600;
        font-size: 0.8rem;
        margin-left: 6px;
    }
    .item-title {
        color: #2C2C2C;
        font-weight: 700;
        font-size: 0.98rem;
        line-height: 1.3;
    }
    .item-desc {
        color: #525252;
        font-size: 0.84rem;
        margin-top: 4px;
        line-height: 1.4;
    }

    /* File uploader & Camera container */
    div[data-testid="stFileUploader"],
    div[data-testid="stCameraInput"] {
        background-color: #FFFFFF;
        border: 1px dashed #C7C7C7;
        border-radius: 8px;
        padding: 8px;
    }

    /* Thumbnail image */
    .thumb-preview img {
        border: 1px solid #C7C7C7;
        border-radius: 6px;
    }

    /* Info / Alert Box Styling with Theme Palette */
    div[data-testid="stAlert"] {
        background-color: #E5E5E5 !important;
        border: 1px solid #C7C7C7 !important;
        border-left: 5px solid #525252 !important;
        color: #2C2C2C !important;
        border-radius: 8px !important;
    }
    div[data-testid="stAlert"] p {
        color: #2C2C2C !important;
        font-size: 0.92rem !important;
    }

    /* Custom Scrollbar for Containers */
    div[data-testid="stContainer"]::-webkit-scrollbar,
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    div[data-testid="stContainer"]::-webkit-scrollbar-track,
    ::-webkit-scrollbar-track {
        background: #E5E5E5;
        border-radius: 4px;
    }
    div[data-testid="stContainer"]::-webkit-scrollbar-thumb,
    ::-webkit-scrollbar-thumb {
        background: #C7C7C7;
        border-radius: 4px;
    }
    div[data-testid="stContainer"]::-webkit-scrollbar-thumb:hover,
    ::-webkit-scrollbar-thumb:hover {
        background: #E81D23;
    }

    /* Search Input */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        border: 1px solid #C7C7C7 !important;
        border-radius: 6px !important;
        color: #2C2C2C !important;
        font-size: 0.88rem !important;
        padding: 6px 12px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #E81D23 !important;
        box-shadow: 0 0 0 1px #E81D23 !important;
    }

    /* Compact Row Container Tweaks */
    .compact-row-box [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 5px 8px !important;
        margin-bottom: 4px !important;
    }

    /* iPad & Mobile Touch Target Enhancements */
    /* Tabs for iPad & Mobile */
    div[data-testid="stTabs"] button[role="tab"] {
        padding: 12px 20px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #525252 !important;
        min-height: 48px !important;
        border-radius: 8px 8px 0 0 !important;
        touch-action: manipulation;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #E81D23 !important;
        border-bottom: 3px solid #E81D23 !important;
        background-color: #FAFAFA !important;
    }

    /* Touch targets for Checkbox */
    div[data-testid="stCheckbox"] {
        min-width: 44px !important;
        min-height: 44px !important;
    }
    div[data-testid="stCheckbox"] label span[role="checkbox"] {
        width: 22px !important;
        height: 22px !important;
        border-radius: 6px !important;
        border: 2px solid #C7C7C7 !important;
    }
    div[data-testid="stCheckbox"] label span[role="checkbox"][aria-checked="true"] {
        background-color: #E81D23 !important;
        border-color: #E81D23 !important;
    }

    /* Action Buttons Touch Target */
    button[data-testid^="baseButton"] {
        min-height: 44px !important;
        font-size: 14px !important;
        touch-action: manipulation;
    }

    /* Delete Button */
    button[key^="del_"] {
        min-width: 40px !important;
        min-height: 40px !important;
    }

    /* Radio Group Touch Targets */
    div[data-testid="stRadio"] > div {
        flex-wrap: wrap !important;
        gap: 8px !important;
    }
    div[data-testid="stRadio"] label {
        min-height: 38px !important;
        padding: 6px 12px !important;
        touch-action: manipulation;
    }

    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #E81D23 !important;
    }

    /* ========================================================= */
    /* MOBILE FIX: PREVENT VERTICAL COLUMN BREAKING INSIDE CARDS */
    /* ========================================================= */
    /* Item Card Container (Nested inside checklist scroll container) */
    div[data-testid="stContainer"] div[data-testid="stContainer"] {
        padding: 6px 8px !important;
        margin-bottom: 6px !important;
        border-color: #C7C7C7 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
    }

    /* Force the card's horizontal columns to remain strictly in 1 single horizontal row */
    div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stHorizontalBlock"],
    div[data-testid="stContainer"] div[data-testid="stContainer"] .stHorizontalBlock {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 6px !important;
        width: 100% !important;
    }

    /* Override min-width and flex on all 4 columns inside the item card */
    div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"],
    div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn {
        min-width: 0 !important;
        margin-bottom: 0 !important;
    }

    /* Col 1: Checkbox */
    div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"]:nth-of-type(1),
    div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn:nth-of-type(1) {
        width: 36px !important;
        min-width: 36px !important;
        max-width: 36px !important;
        flex: 0 0 36px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    /* Col 2: Image Thumbnail */
    div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"]:nth-of-type(2),
    div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn:nth-of-type(2) {
        width: 44px !important;
        min-width: 44px !important;
        max-width: 44px !important;
        flex: 0 0 44px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    /* Col 3: Text content (title, category, score, desc) */
    div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"]:nth-of-type(3),
    div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn:nth-of-type(3) {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
        overflow: hidden !important;
    }

    /* Col 4: Delete button */
    div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"]:nth-of-type(4),
    div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn:nth-of-type(4) {
        width: 38px !important;
        min-width: 38px !important;
        max-width: 38px !important;
        flex: 0 0 38px !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    /* Compact Delete Button in Card */
    div[data-testid="stContainer"] div[data-testid="stContainer"] button {
        min-width: 32px !important;
        width: 34px !important;
        height: 34px !important;
        min-height: 34px !important;
        padding: 0 !important;
        font-size: 13px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 6px !important;
    }

    /* Top Action Buttons row: keep in 1 row on mobile */
    .top-action-bar .stHorizontalBlock,
    .top-action-bar [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 6px !important;
        width: 100% !important;
    }
    .top-action-bar .stColumn,
    .top-action-bar [data-testid="stColumn"],
    .top-action-bar [data-testid="column"],
    .top-action-bar [data-testid="stHorizontalBlock"] > div {
        flex: 1 1 33.33% !important;
        min-width: 0 !important;
        width: 33.33% !important;
    }
    .top-action-bar button {
        padding: 8px 4px !important;
        font-size: 0.82rem !important;
    }

    /* Responsive adjustments specifically for Mobile Viewports (< 768px) */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.8rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        /* Item card padding on mobile */
        div[data-testid="stContainer"] div[data-testid="stContainer"] {
            padding: 5px 6px !important;
            margin-bottom: 4px !important;
        }
        /* Reinforce nowrap on mobile inside cards */
        div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stHorizontalBlock"],
        div[data-testid="stContainer"] div[data-testid="stContainer"] .stHorizontalBlock {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 4px !important;
        }
        div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"],
        div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn {
            min-width: 0 !important;
        }
        div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"]:nth-of-type(1),
        div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn:nth-of-type(1) {
            width: 34px !important;
            min-width: 34px !important;
            max-width: 34px !important;
            flex: 0 0 34px !important;
        }
        div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"]:nth-of-type(2),
        div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn:nth-of-type(2) {
            width: 42px !important;
            min-width: 42px !important;
            max-width: 42px !important;
            flex: 0 0 42px !important;
        }
        div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"]:nth-of-type(3),
        div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn:nth-of-type(3) {
            flex: 1 1 auto !important;
            min-width: 0 !important;
            width: auto !important;
        }
        div[data-testid="stContainer"] div[data-testid="stContainer"] div[data-testid="stColumn"]:nth-of-type(4),
        div[data-testid="stContainer"] div[data-testid="stContainer"] .stColumn:nth-of-type(4) {
            width: 36px !important;
            min-width: 36px !important;
            max-width: 36px !important;
            flex: 0 0 36px !important;
        }
    }

    /* Processing Radar Animation & Modal Styling */
    @keyframes radar-sweep {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(232, 29, 35, 0.45); }
        70% { box-shadow: 0 0 0 16px rgba(232, 29, 35, 0); }
        100% { box-shadow: 0 0 0 0 rgba(232, 29, 35, 0); }
    }
    .radar-scan-anim {
        width: 76px;
        height: 76px;
        border-radius: 50%;
        border: 3px solid #E81D23;
        background: radial-gradient(circle, rgba(232,29,35,0.12) 0%, rgba(229,229,229,0.4) 100%);
        margin: 0 auto;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: pulse-ring 2s infinite;
    }
    .radar-beam {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: conic-gradient(from 0deg, rgba(232, 29, 35, 0.45) 0deg, transparent 65deg);
        animation: radar-sweep 2s linear infinite;
    }
    .radar-icon {
        font-size: 30px;
        z-index: 2;
    }

    /* Modal / Dialog Styling */
    div[role="dialog"] {
        border-radius: 12px !important;
        border: 2px solid #C7C7C7 !important;
        box-shadow: 0 8px 30px rgba(44, 44, 44, 0.2) !important;
    }
    div[role="dialog"] h2 {
        color: #2C2C2C !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        border-bottom: 1px solid #E5E5E5 !important;
        padding-bottom: 8px !important;
    }

    /* Custom Navigation Bar - Keep 3 tabs side by side on mobile & desktop */
    .custom-nav-bar {
        margin-bottom: 14px;
    }
    div[data-testid="stHorizontalBlock"]:has(button[key="nav_btn_scan"]) {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 8px !important;
        width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(button[key="nav_btn_scan"]) > div[data-testid="stColumn"],
    div[data-testid="stHorizontalBlock"]:has(button[key="nav_btn_scan"]) > .stColumn {
        flex: 1 1 33.33% !important;
        min-width: 0 !important;
        width: 33.33% !important;
        max-width: 33.33% !important;
    }
    .custom-nav-bar button,
    button[key="nav_btn_scan"],
    button[key="nav_btn_check"],
    button[key="nav_btn_tray"] {
        height: 48px !important;
        min-height: 48px !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        touch-action: manipulation;
        white-space: nowrap !important;
    }
    button[key="nav_btn_scan"][kind="primary"],
    button[key="nav_btn_check"][kind="primary"],
    button[key="nav_btn_tray"][kind="primary"] {
        background-color: #E81D23 !important;
        color: #FFFFFF !important;
        border: 2px solid #E81D23 !important;
        box-shadow: 0 2px 8px rgba(232, 29, 35, 0.25) !important;
    }
    button[key="nav_btn_scan"][kind="secondary"],
    button[key="nav_btn_check"][kind="secondary"],
    button[key="nav_btn_tray"][kind="secondary"] {
        background-color: #E5E5E5 !important;
        color: #525252 !important;
        border: 1px solid #C7C7C7 !important;
    }
    button[key="nav_btn_scan"][kind="secondary"]:hover,
    button[key="nav_btn_check"][kind="secondary"]:hover,
    button[key="nav_btn_tray"][kind="secondary"]:hover {
        background-color: #FFFFFF !important;
        color: #2C2C2C !important;
        border-color: #525252 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. เตรียมตัวแปรความจำ (Session State) ---
if 'detected_list' not in st.session_state:
    st.session_state.detected_list = []

if 'active_view' not in st.session_state:
    st.session_state.active_view = "scan"

if 'tray_check_results' not in st.session_state:
    st.session_state.tray_check_results = None

if 'tray_check_data' not in st.session_state:
    st.session_state.tray_check_data = None

# ==========================================
# ฟังก์ชันจัดการเมื่อมีการเปลี่ยนรูปภาพ
# ==========================================
def keep_only_checked_items():
    kept_items = [item for item in st.session_state.detected_list if item.get('checked', False)]
    st.session_state.detected_list = kept_items
    
    keys_to_del = [k for k in st.session_state.keys() if str(k).startswith("chk_")]
    for k in keys_to_del:
        del st.session_state[k]

# ==========================================

@st.cache_resource
def load_scanner():
    if not os.path.exists('mock_database'): os.makedirs('mock_database')
    return ShapeScanner()

def get_product_info(filename):
    try:
        with open('mock_database/data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item['filename'] == os.path.basename(filename):
                    return item
    except: return None
    return None

scanner = load_scanner()

# Header Section with Palette Styling
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; background: #FFFFFF; border-bottom: 2px solid #C7C7C7; padding-bottom: 12px; margin-bottom: 15px; gap: 10px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="background-color: #E81D23; color: #FFFFFF; width: 42px; height: 42px; display: flex; align-items: center; justify-content: center; border-radius: 8px; font-weight: bold; box-shadow: 0 2px 6px rgba(232, 29, 35, 0.25);">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
        </div>
        <div>
            <h1 style="margin: 0; color: #2C2C2C; font-size: 22px; font-weight: 700; letter-spacing: -0.3px;">ระบบเช็คลิสต์เครื่องมือ (Multi-Object)</h1>
            <p style="margin: 2px 0 0 0; color: #525252; font-size: 12px;">SIFT Feature Matching & Real-time Inspection</p>
        </div>
    </div>
    <div style="display: flex; gap: 6px; align-items: center;">
        <span style="background-color: #E5E5E5; color: #2C2C2C; border: 1px solid #C7C7C7; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 6px;">iPad/Mobile Ready</span>
        <span style="background-color: #E81D23; color: #FFFFFF; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 6px;">READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# Popup ประมวลผลและเปลี่ยนหน้าอัตโนมัติ
# ==========================================
@st.dialog("กำลังประมวลผลสแกนเครื่องมือ", width="small")
def scan_processing_dialog(opencv_img):
    st.markdown("""
    <div style="text-align: center; padding: 4px 0 8px 0;">
        <div class="radar-scan-anim">
            <div class="radar-beam"></div>
            <div class="radar-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E81D23" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24"/><path d="m14.83 9.17 4.24-4.24"/><path d="m14.83 14.83 4.24 4.24"/><path d="m9.17 14.83-4.24 4.24"/></svg>
            </div>
        </div>
        <h4 style="color: #2C2C2C; margin: 12px 0 4px 0; font-size: 1.06rem; font-weight: 700;">กำลังวิเคราะห์และตรวจจับเครื่องมือ</h4>
        <p style="color: #525252; font-size: 0.82rem; margin-bottom: 6px;">ระบบกำลังใช้ SIFT Feature Matching & Tiling Inspection</p>
    </div>
    """, unsafe_allow_html=True)
    
    status_text = st.empty()
    prog_bar = st.progress(20)
    
    status_text.markdown("<p style='text-align: center; color: #525252; font-size: 0.85rem;'>ขั้นที่ 1/3: ค้นหาจุดเด่นในภาพ (Extracting Keypoints)...</p>", unsafe_allow_html=True)
    time.sleep(0.35)
    
    prog_bar.progress(55)
    status_text.markdown("<p style='text-align: center; color: #525252; font-size: 0.85rem;'>ขั้นที่ 2/3: เปรียบเทียบกับฐานข้อมูลอุปกรณ์ (Feature Matching)...</p>", unsafe_allow_html=True)
    
    # รันการค้นหาอุปกรณ์จริง
    results = scanner.scan_with_tiling(opencv_img, threshold=8)
    
    prog_bar.progress(90)
    status_text.markdown("<p style='text-align: center; color: #525252; font-size: 0.85rem;'>ขั้นที่ 3/3: ประมวลผลและสรุปรายการ...</p>", unsafe_allow_html=True)
    time.sleep(0.25)
    prog_bar.progress(100)
    
    if results:
        count_new = 0
        existing_files = [x['filename'] for x in st.session_state.detected_list]
        
        for res in results:
            if res['filename'] not in existing_files:
                info = get_product_info(res['filename'])
                display_name = info['name'] if info else res['filename']
                category = info['category'] if info else "Unknown"
                description = info['description'] if info else "-"
                
                st.session_state.detected_list.append({
                    "filename": res['filename'],
                    "name": display_name,
                    "category": category,
                    "description": description,
                    "score": res['score'],
                    "checked": False
                })
                count_new += 1
        
        if count_new > 0:
            status_text.markdown(f"""
            <div style="background-color: #FFFFFF; border: 2px solid #E81D23; border-radius: 8px; padding: 12px; margin: 10px 0; text-align: center;">
                <strong style="color: #2C2C2C; font-size: 1.02rem; display: block; margin-top: 4px;">ตรวจพบอุปกรณ์ {count_new} รายการใหม่</strong>
                <span style="color: #525252; font-size: 0.84rem; display: block; margin-top: 2px;">กำลังนำคุณไปยังหน้าแสดงผลรายการตรวจสอบ...</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            status_text.markdown(f"""
            <div style="background-color: #E5E5E5; border: 1px solid #C7C7C7; border-left: 4px solid #525252; border-radius: 8px; padding: 12px; margin: 10px 0; text-align: center;">
                <strong style="color: #2C2C2C; font-size: 0.96rem; display: block;">ตรวจพบ {len(results)} รายการ (มีอยู่ในรายการแล้ว)</strong>
                <span style="color: #525252; font-size: 0.84rem; display: block; margin-top: 2px;">กำลังนำคุณไปยังหน้าแสดงผลรายการตรวจสอบ...</span>
            </div>
            """, unsafe_allow_html=True)
            
        time.sleep(1.0)
        st.session_state.active_view = "checklist"
        st.rerun()
    else:
        status_text.markdown("""
        <div style="background-color: #FFFFFF; border: 1px solid #E81D23; border-radius: 8px; padding: 12px; margin: 10px 0; text-align: center;">
            <strong style="color: #E81D23; font-size: 0.95rem; display: block;">ไม่พบอุปกรณ์ที่ตรงกับฐานข้อมูล</strong>
            <span style="color: #525252; font-size: 0.82rem; display: block; margin-top: 4px;">คำแนะนำ: ปรับแสงสว่าง หรือวางอุปกรณ์ให้เห็นลายเส้นชัดเจน</span>
        </div>
        """, unsafe_allow_html=True)
        
        c_dlg1, c_dlg2 = st.columns(2)
        with c_dlg1:
            if st.button("ลองสแกนใหม่", use_container_width=True):
                st.rerun()
        with c_dlg2:
            if st.button("ไปยังรายการ", use_container_width=True):
                st.session_state.active_view = "checklist"
                st.rerun()

# ==========================================
# ฟังก์ชันส่วนสแกนเครื่องมือ (Scanner)
# ==========================================
def render_scanner():
    st.markdown("""
    <div style="border-left: 4px solid #E81D23; padding-left: 10px; margin-bottom: 10px;">
        <h3 style="margin: 0; color: #2C2C2C; font-size: 18px;">สแกนเครื่องมือ</h3>
    </div>
    """, unsafe_allow_html=True)
    
    input_method = st.radio("เลือกวิธี:", ["Camera", "Upload Image"], horizontal=True, key="scan_input_method")
    
    opencv_img = None
    
    if input_method == "Camera":
        st.caption("คำแนะนำ: ถือ iPad หรือมือถือขนานกับโต๊ะ วางเครื่องมือไม่ซ้อนทับกัน")
        img_file = st.camera_input("ถ่ายภาพเครื่องมือ", key="cam_input", on_change=keep_only_checked_items)
        if img_file:
            opencv_img = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    elif input_method == "Upload Image":
        img_file = st.file_uploader("อัปโหลดภาพเครื่องมือ", type=['jpg','png'], key="file_input", on_change=keep_only_checked_items)
        if img_file:
            opencv_img = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    if opencv_img is not None:
        viz_img = None
        kp_count = 0
        try:
            viz_img, kp_count = scanner.visualize_keypoints(opencv_img.copy())
        except AttributeError:
            viz_img = opencv_img
            kp_count = "N/A"

        st.write("---") 
        img_col1, img_col2 = st.columns(2) 

        with img_col1:
            st.image(opencv_img, channels="BGR", caption="ภาพถ่าย", use_container_width=True)

        with img_col2:
            caption_text = f"จุดสแกน (พบ {kp_count} จุด)"
            st.image(viz_img, channels="BGR", caption=caption_text, use_container_width=True)
        
        if isinstance(kp_count, int) and kp_count < 500:
             st.warning(f"พบจุดเด่นน้อย ({kp_count}) แนะนำให้ปรับแสงหรือขยับเข้าใกล้เครื่องมือ")
        
        st.divider()

        if st.button("สแกนหาอุปกรณ์ทั้งหมด", type="primary", use_container_width=True):
            scan_processing_dialog(opencv_img)

# ==========================================
# ฟังก์ชันส่วนรายการตรวจสอบ (Checklist)
# ==========================================
def render_checklist():
    total_count = len(st.session_state.detected_list)
    checked_count = sum(1 for x in st.session_state.detected_list if x.get('checked', False))
    remaining_count = total_count - checked_count
    percent = int((checked_count / total_count * 100)) if total_count > 0 else 0

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; border-left: 4px solid #E81D23; padding-left: 10px; margin-bottom: 10px;">
        <h3 style="margin: 0; color: #2C2C2C; font-size: 18px;">รายการตรวจสอบ</h3>
        <span style="background-color: #E81D23; color: #FFFFFF; padding: 2px 10px; border-radius: 12px; font-size: 13px; font-weight: 600;">{total_count} รายการ</span>
    </div>
    """, unsafe_allow_html=True)
    
    # --- ปุ่มจัดการขนาดใหญ่แตะสะดวก ---
    st.markdown("<div class='top-action-bar'>", unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns(3)
    
    def update_all_checked(value):
        for i in range(len(st.session_state.detected_list)):
            st.session_state.detected_list[i]['checked'] = value
            if f"chk_{i}" in st.session_state:
                st.session_state[f"chk_{i}"] = value

    with c_btn1:
        if st.button("ตรวจครบ", use_container_width=True):
            update_all_checked(True)
            st.rerun()
            
    with c_btn2:
        if st.button("ยกเลิก", use_container_width=True):
            update_all_checked(False)
            st.rerun()
            
    with c_btn3:
        if st.button("ล้างหมด", type="primary", use_container_width=True):
            st.session_state.detected_list = []
            for k in list(st.session_state.keys()):
                if str(k).startswith("chk_"):
                    del st.session_state[k]
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
            
    if total_count > 0:
        # แถบสรุปสถานะย่อ
        st.markdown(f"""
        <div style="display: flex; gap: 8px; margin: 8px 0;">
            <div style="flex: 1; background: #2C2C2C; color: #FFFFFF; border-radius: 6px; padding: 6px 8px; text-align: center;">
                <div style="font-size: 11px; color: #C7C7C7;">ทั้งหมด</div>
                <div style="font-size: 16px; font-weight: 700;">{total_count}</div>
            </div>
            <div style="flex: 1; background: #E5E5E5; color: #2C2C2C; border: 1px solid #C7C7C7; border-radius: 6px; padding: 6px 8px; text-align: center;">
                <div style="font-size: 11px; color: #525252;">ยังไม่ตรวจ</div>
                <div style="font-size: 16px; font-weight: 700; color: #E81D23;">{remaining_count}</div>
            </div>
            <div style="flex: 1; background: #E5E5E5; color: #2C2C2C; border: 1px solid #C7C7C7; border-radius: 6px; padding: 6px 8px; text-align: center;">
                <div style="font-size: 11px; color: #525252;">ตรวจแล้ว</div>
                <div style="font-size: 16px; font-weight: 700; color: #2C2C2C;">{checked_count} ({percent}%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar
        st.progress(percent / 100)
        
        # แจ้งเตือนเมื่อตรวจครบ 100%
        if checked_count == total_count and total_count > 0:
            st.markdown("""
            <div style="background-color: #FFFFFF; border: 2px solid #E81D23; border-radius: 8px; padding: 10px; margin: 8px 0; text-align: center;">
                <strong style="color: #2C2C2C; font-size: 15px;">ตรวจสอบอุปกรณ์ครบทุกชิ้นเรียบร้อย 100% (Ready for Handover)</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # ฟิลเตอร์และค้นหา
        c_filter, c_view = st.columns([1.3, 1])
        with c_filter:
            filter_mode = st.radio(
                "กรองสถานะ:",
                options=["all", "unchecked", "checked"],
                format_func=lambda x: f"ทั้งหมด ({total_count})" if x == "all" else (f"ยังไม่ตรวจ ({remaining_count})" if x == "unchecked" else f"ตรวจแล้ว ({checked_count})"),
                horizontal=True,
                label_visibility="collapsed",
                key="chk_filter_radio"
            )
        with c_view:
            view_mode = st.radio(
                "รูปแบบ:",
                ["กะทัดรัด (Compact)", "การ์ด 2 คอลัมน์ (Grid)"],
                horizontal=True,
                label_visibility="collapsed",
                key="chk_view_radio"
            )
            
        search_kw = st.text_input("ค้นหา", placeholder="พิมพ์ชื่อ หรือหมวดหมู่...", label_visibility="collapsed", key="chk_search_input")
    else:
        filter_mode = "all"
        view_mode = "กะทัดรัด (Compact)"
        search_kw = ""

    st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

    # --- ส่วนแสดงรายการ (Scrollable Box) ---
    with st.container(height=480):
        if not st.session_state.detected_list:
            st.info("ยังไม่มีรายการ... กรุณาถ่ายภาพหรืออัปโหลดทางฝั่งสแกน")
        else:
            filtered_indices = []
            for i, item in enumerate(st.session_state.detected_list):
                if search_kw:
                    kw = search_kw.lower()
                    if kw not in item['name'].lower() and kw not in item.get('category', '').lower():
                        continue
                if filter_mode == "unchecked" and item.get('checked', False):
                    continue
                if filter_mode == "checked" and not item.get('checked', False):
                    continue
                filtered_indices.append(i)

            if not filtered_indices:
                st.markdown("<p style='text-align: center; color: #525252; padding: 20px 0;'>ไม่พบรายการที่ตรงกับเงื่อนไข</p>", unsafe_allow_html=True)
            elif "2 คอลัมน์" in view_mode:
                for row_idx in range(0, len(filtered_indices), 2):
                    pair_cols = st.columns(2)
                    for col_idx, pair_col in enumerate(pair_cols):
                        item_sub_idx = row_idx + col_idx
                        if item_sub_idx < len(filtered_indices):
                            orig_i = filtered_indices[item_sub_idx]
                            item = st.session_state.detected_list[orig_i]
                            is_item_checked = item.get('checked', False)
                            
                            with pair_col:
                                with st.container(border=True):
                                    gc1, gc2, gc3 = st.columns([0.15, 0.71, 0.14])
                                    with gc1:
                                        is_checked = st.checkbox(
                                            label=f"เลือก {item['name']}",
                                            value=is_item_checked,
                                            key=f"chk_{orig_i}",
                                            label_visibility="collapsed"
                                        )
                                        st.session_state.detected_list[orig_i]['checked'] = is_checked
                                    with gc2:
                                        name_style = "color: #525252; text-decoration: line-through;" if is_item_checked else "color: #2C2C2C; font-weight: 600;"
                                        st.markdown(f"""
                                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                            <div style="font-size: 0.88rem; {name_style} line-height: 1.2;">
                                                {item['name']}
                                            </div>
                                            <span class='badge-score'>★ {item['score']}</span>
                                        </div>
                                        <div style="margin-top: 3px;">
                                            <span class='badge-category' style="font-size: 0.70rem; padding: 1px 6px;">{item['category']}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with gc3:
                                        if st.button("✕", key=f"del_grid_{orig_i}", help="ลบรายการนี้"):
                                            st.session_state.detected_list.pop(orig_i)
                                            for k in list(st.session_state.keys()):
                                                if str(k).startswith("chk_"):
                                                    del st.session_state[k]
                                            st.rerun()
            else:
                for orig_i in filtered_indices:
                    item = st.session_state.detected_list[orig_i]
                    is_item_checked = item.get('checked', False)
                    
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([0.10, 0.14, 0.66, 0.10])
                        
                        with c1:
                            is_checked = st.checkbox(
                                label=f"เลือก {item['name']}", 
                                value=is_item_checked, 
                                key=f"chk_{orig_i}",
                                label_visibility="collapsed"
                            )
                            st.session_state.detected_list[orig_i]['checked'] = is_checked
                        
                        with c2:
                            try:
                                st.image(f"mock_database/{item['filename']}", width=40)
                            except: 
                                st.write("")
                        
                        with c3:
                            name_color = "color: #525252; text-decoration: line-through;" if is_item_checked else "color: #2C2C2C; font-weight: 600;"
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 4px; overflow: hidden;">
                                <span style="font-size: 0.88rem; {name_color} white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    {item['name']}
                                </span>
                                <span class='badge-score' style="font-size: 0.74rem; flex-shrink: 0;">★ {item['score']}</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 6px; margin-top: 2px; overflow: hidden;">
                                <span class='badge-category' style="font-size: 0.68rem; padding: 1px 6px; flex-shrink: 0;">{item['category']}</span>
                                <span class='item-desc' style="font-size: 0.74rem; color: #525252; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    {item['description']}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with c4:
                            if st.button("✕", key=f"del_{orig_i}", help="ลบรายการนี้"):
                                st.session_state.detected_list.pop(orig_i)
                                for k in list(st.session_state.keys()):
                                    if str(k).startswith("chk_"):
                                        del st.session_state[k]
                                st.rerun()


# ==========================================
# ฟังก์ชันส่วนจัดการถาด (Tray Template)
# ==========================================
def render_tray_register():
    """รับลงทะเบียน Template ถาด: ถ่ายภาพถาดเต็ม → บันทึกตำแหน่งและ Brightness"""
    st.markdown("""
    <div style="background:#E5E5E5;border:1px solid #C7C7C7;border-left:4px solid #2C2C2C;
                border-radius:8px;padding:10px 12px;margin-bottom:12px;">
        <p style="margin:0;color:#2C2C2C;font-size:0.88rem;font-weight:600;">\U0001f4cb วิธีใช้:</p>
        <p style="margin:4px 0 0 0;color:#525252;font-size:0.82rem;">
            1. ใส่เครื่องมือในถาดให้ครบทุกชิ้น<br>
            2. ถ่ายภาพจากมุมตั้งฉากกับถาด (Top-down) ให้เห็นถาดทั้งใบ<br>
            3. ใส่ชื่อถาด แล้วกด “ลงทะเบียน”
        </p>
    </div>
    """, unsafe_allow_html=True)

    tray_name = st.text_input(
        "ชื่อถาด",
        placeholder="เช่น YA 1/2, YA 2/2, BA",
        key="tray_reg_name"
    )

    input_method = st.radio("เลือกวิธี:", ["Camera", "Upload Image"],
                            horizontal=True, key="tray_reg_method")
    opencv_img = None
    if input_method == "Camera":
        img_file = st.camera_input("ถ่ายภาพถาดเต็ม", key="tray_reg_cam")
    else:
        img_file = st.file_uploader("อัปโหลดภาพถาด", type=['jpg', 'png', 'jpeg'],
                                    key="tray_reg_file")

    if img_file:
        opencv_img = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        st.image(opencv_img, channels="BGR", caption="ภาพถาดที่จะลงทะเบียน", use_container_width=True)

    if opencv_img is not None:
        if not tray_name.strip():
            st.warning("⚠️ กรุณาใส่ชื่อถาดก่อนลงทะเบียน")
        else:
            if st.button("\U0001f5c2️ ลงทะเบียน Template",
                         type="primary", use_container_width=True, key="tray_reg_btn"):
                tray_id = (tray_name.strip()
                           .replace(" ", "_").replace("/", "_").replace(".", "_"))
                with st.spinner(f"กำลังวิเคราะห์ตำแหน่งเครื่องมือในถาด '{tray_name}'..."):
                    tray_data = scanner.register_tray_template(
                        opencv_img, tray_id, tray_name.strip()
                    )

                slots = tray_data.get('slots', [])
                if slots:
                    st.markdown(f"""
                    <div style="background:#FFFFFF;border:2px solid #E81D23;border-radius:8px;
                                padding:10px 14px;margin:8px 0;text-align:center;">
                        <strong style="color:#2C2C2C;font-size:1rem;">
                            ลงทะเบียนถาด '{tray_name}' สำเร็จ!
                        </strong>
                        <span style="color:#525252;font-size:0.84rem;display:block;margin-top:2px;">
                            พบและบันทึก {len(slots)} รายการ
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.container(height=200):
                        for slot in slots:
                            st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:8px;
                                        padding:4px 0;border-bottom:1px solid #E5E5E5;">
                                <span style="color:#2C2C2C;font-size:0.85rem;">
                                    \u2705 <strong>{slot['name']}</strong>
                                </span>
                                <span style="background:#E5E5E5;color:#525252;font-size:0.72rem;
                                             padding:1px 6px;border-radius:8px;">
                                    {slot.get('category', 'เครื่องมือ')}
                                </span>
                                <span style="color:#C7C7C7;font-size:0.70rem;margin-left:auto;">
                                    ☉ {slot.get('mean_brightness', 110.0):.0f}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("ไม่พบเครื่องมือในภาพ กรุณาปรับแสงและลองใหม่")


def render_tray_check():
    """ตรวจสอบถาดว่าเครื่องมือครบหรือขาด"""
    templates = scanner.list_tray_templates()
    if not templates:
        st.info("ยังไม่มีถาดที่ลงทะเบียน กรุณาไปที่แท็บ 'ลงทะเบียนถาด' ก่อน")
        return

    # Dropdown เลือกถาด
    tray_options = {
        f"{t['tray_name']} ({t['slot_count']} รายการ)": t['tray_id']
        for t in templates
    }
    selected_label = st.selectbox("เลือกถาด",
                                   options=list(tray_options.keys()),
                                   key="tray_check_select")
    selected_id = tray_options[selected_label]

    sel_tray = next((t for t in templates if t['tray_id'] == selected_id), None)
    if sel_tray:
        reg_at = sel_tray.get('registered_at', '-')[:10]
        st.caption(f"ลงทะเบียนเมื่อ: {reg_at}")

    st.divider()

    input_method = st.radio("เลือกวิธี:", ["Camera", "Upload Image"],
                            horizontal=True, key="tray_check_method")
    opencv_img = None
    if input_method == "Camera":
        img_file = st.camera_input("ถ่ายภาพถาดปัจจุบัน", key="tray_check_cam")
    else:
        img_file = st.file_uploader("อัปโหลดภาพถาดปัจจุบัน",
                                    type=['jpg', 'png', 'jpeg'], key="tray_check_file")

    if img_file:
        opencv_img = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        st.image(opencv_img, channels="BGR", caption="ภาพถาดปัจจุบัน", use_container_width=True)

    if opencv_img is not None:
        if st.button("\U0001f50d ตรวจสอบถาด",
                     type="primary", use_container_width=True, key="tray_check_btn"):
            with st.spinner("กำลังเปรียบเทียบตำแหน่งเครื่องมือ..."):
                results, tray_info = scanner.check_tray_slots(opencv_img, selected_id)
            if results is None:
                st.error(f"เกิดข้อผิดพลาด: {tray_info}")
            else:
                st.session_state.tray_check_results = results
                st.session_state.tray_check_data = tray_info
                st.rerun()

    # แสดงผลการตรวจสอบ (ถ้ามี และตรงกับถาดที่เลือกอยู่)
    r_cache = st.session_state.get('tray_check_results')
    d_cache = st.session_state.get('tray_check_data')
    if r_cache and d_cache and d_cache.get('tray_id') == selected_id:
        results  = r_cache
        tray_info = d_cache

        present  = [r for r in results if r['status'] == 'present']
        missing  = [r for r in results if r['status'] == 'missing']
        pct      = int(len(present) / len(results) * 100) if results else 0

        st.markdown(f"""
        <div style="display:flex;gap:8px;margin:10px 0 6px 0;">
            <div style="flex:1;background:#2C2C2C;color:#FFFFFF;border-radius:6px;padding:8px;text-align:center;">
                <div style="font-size:10px;color:#C7C7C7;">ทั้งหมด</div>
                <div style="font-size:20px;font-weight:700;">{len(results)}</div>
            </div>
            <div style="flex:1;background:#E5E5E5;color:#2C2C2C;border:1px solid #C7C7C7;
                        border-radius:6px;padding:8px;text-align:center;">
                <div style="font-size:10px;color:#525252;">ครบ</div>
                <div style="font-size:20px;font-weight:700;">{len(present)}</div>
            </div>
            <div style="flex:1;background:{'#FFFFFF' if missing else '#E5E5E5'};
                        border:{'2px solid #E81D23' if missing else '1px solid #C7C7C7'};
                        border-radius:6px;padding:8px;text-align:center;">
                <div style="font-size:10px;color:{'#E81D23' if missing else '#525252'};">ขาด</div>
                <div style="font-size:20px;font-weight:700;color:{'#E81D23' if missing else '#2C2C2C'};"
                >{len(missing)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(pct / 100)

        if not missing:
            st.markdown("""
            <div style="background:#FFFFFF;border:2px solid #2C2C2C;border-radius:8px;
                        padding:10px;text-align:center;margin:6px 0;">
                <strong style="color:#2C2C2C;">✅ เครื่องมือครบทุกชิ้น!</strong>
            </div>
            """, unsafe_allow_html=True)

        # รายการแต่ละชิ้น
        with st.container(height=280):
            for r in results:
                is_p  = r['status'] == 'present'
                icon  = "✅" if is_p else ("❌" if r['status'] == 'missing' else "❓")
                bg    = "#FFFFFF" if is_p else ("#FFF0F0" if r['status'] == 'missing' else "#FAFAFA")
                bdr   = "#E5E5E5" if is_p else ("#E81D23" if r['status'] == 'missing' else "#C7C7C7")
                nstyl = "color:#525252;" if is_p else "color:#2C2C2C;font-weight:700;"
                dpct  = r.get('brightness_diff_pct', 0)
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {bdr};border-radius:6px;
                     padding:6px 10px;margin-bottom:4px;display:flex;align-items:center;gap:8px;">
                    <span style="font-size:1.05rem;flex-shrink:0;">{icon}</span>
                    <span style="{nstyl}font-size:0.86rem;flex:1;overflow:hidden;
                                 text-overflow:ellipsis;white-space:nowrap;">{r['name']}</span>
                    <span style="background:#E5E5E5;color:#525252;font-size:0.68rem;
                         padding:1px 6px;border-radius:8px;flex-shrink:0;">{r['category']}</span>
                    <span style="font-size:0.68rem;color:#C7C7C7;flex-shrink:0;">Δ{dpct:.0f}%</span>
                </div>
                """, unsafe_allow_html=True)

        # ปุ่มเพิ่มรายการขาดเข้า Checklist
        if missing:
            if st.button(
                f"\u2795 เพิ่ม {len(missing)} รายการขาดเข้า Checklist",
                use_container_width=True, type="primary", key="tray_add_missing_btn"
            ):
                existing_files = [x['filename'] for x in st.session_state.detected_list]
                added = 0
                for r in missing:
                    if r['filename'] not in existing_files:
                        st.session_state.detected_list.append({
                            "filename":    r['filename'],
                            "name":        r['name'],
                            "category":    r['category'],
                            "description": r.get('description', '-'),
                            "score":       r.get('score', 0),
                            "checked":     False
                        })
                        added += 1
                st.session_state.tray_check_results = None
                st.session_state.tray_check_data    = None
                st.session_state.active_view        = "checklist"
                st.rerun()

        # ตัวเลือกเพิ่มเติม
        with st.expander("⚙️ ตัวเลือกเพิ่มเติม"):
            if st.button("ลบ Template นี้",
                         key="tray_delete_btn", type="secondary"):
                scanner.delete_tray_template(selected_id)
                st.session_state.tray_check_results = None
                st.session_state.tray_check_data    = None
                st.success("ลบ Template สำเร็จ")
                st.rerun()


def render_tray():
    """หน้าจัดการถาด Template"""
    st.markdown("""
    <div style="border-left:4px solid #E81D23;padding-left:10px;margin-bottom:10px;">
        <h3 style="margin:0;color:#2C2C2C;font-size:18px;">จัดการถาดเครื่องมือ</h3>
        <p style="margin:2px 0 0 0;color:#525252;font-size:12px;">
            บันทึกตำแหน่งและความสว่างของแต่ละช่อง ตรวจสอบได้ทันที (ไม่ใช้ SIFT)
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_reg, tab_check = st.tabs(["ลงทะเบียนถาด", "ตรวจสอบถาด"])
    with tab_reg:
        render_tray_register()
    with tab_check:
        render_tray_check()


# ==========================================
# เมนูนำทางแบบแท็บ (Interactive Navigation Tabs)
# ==========================================
total_items = len(st.session_state.detected_list)

st.markdown("<div class='custom-nav-bar'>", unsafe_allow_html=True)
c_nav1, c_nav2, c_nav3 = st.columns(3)

with c_nav1:
    is_scan_active = (st.session_state.get('active_view', 'scan') == 'scan')
    btn_type1 = "primary" if is_scan_active else "secondary"
    if st.button("สแกน", key="nav_btn_scan", type=btn_type1, use_container_width=True):
        st.session_state.active_view = "scan"
        st.rerun()

with c_nav2:
    is_check_active = (st.session_state.get('active_view', 'scan') == 'checklist')
    btn_type2 = "primary" if is_check_active else "secondary"
    label_check = f"เช็คลิสต์ ({total_items})"
    if st.button(label_check, key="nav_btn_check", type=btn_type2, use_container_width=True):
        st.session_state.active_view = "checklist"
        st.rerun()

with c_nav3:
    is_tray_active = (st.session_state.get('active_view', 'scan') == 'tray')
    btn_type3 = "primary" if is_tray_active else "secondary"
    tray_count = len(scanner.list_tray_templates())
    label_tray = f"ถาด ({tray_count})"
    if st.button(label_tray, key="nav_btn_tray", type=btn_type3, use_container_width=True):
        st.session_state.active_view = "tray"
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

active_view = st.session_state.get('active_view', 'scan')
if active_view == 'scan':
    render_scanner()
elif active_view == 'checklist':
    render_checklist()
else:
    render_tray()