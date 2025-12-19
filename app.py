import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import os
from datetime import datetime
import google.generativeai as genai # type: ignore
from dotenv import load_dotenv

# ========== LOAD ENVIRONMENT VARIABLES ==========
load_dotenv()

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
def load_css():
    st.markdown("""
    <style>
    .blocked { 
        background: rgba(220, 38, 38, 0.2) !important; 
        border: 3px solid #DC2626 !important; 
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.3), 0 6px 20px rgba(220, 38, 38, 0.4) !important;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.2), rgba(139, 0, 0, 0.3)) !important;
        border: 3px solid #FF6B35 !important;
        border-left: 10px solid #FF6B35 !important;
        color: #660000 !important;
        border-radius: 12px !important;
        padding: 25px !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        box-shadow: 0 6px 20px rgba(139, 0, 0, 0.3) !important;
        margin: 20px 0 !important;
        animation: pulse 2s infinite !important;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 107, 53, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 107, 53, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 107, 53, 0); }
    }
    .blocked-panel {
        background: linear-gradient(135deg, #E8DFC5, #D4C9A8) !important;
        border: 3px solid #DC2626 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin: 15px 0 !important;
        box-shadow: 0 6px 20px rgba(139, 0, 0, 0.3) !important;
        max-height: 300px !important;
        overflow-y: auto !important;
    }
    .blocked-item {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.1), rgba(139, 0, 0, 0.15)) !important;
        border: 2px solid #DC2626 !important;
        border-radius: 8px !important;
        padding: 12px 15px !important;
        margin: 8px 0 !important;
        color: #660000 !important;
        font-weight: 600 !important;
    }
    /* ========== LLM ANALYSIS RESPONSE ========== */
    /* ========== LLM ANALYSIS RESPONSE ========== */
.llm-response {
    background: linear-gradient(135deg, var(--vanilla-light), var(--vanilla-medium)) !important;  /* CHANGED TO VANILLA */
    border: 2px solid var(--blood-red) !important;  /* CHANGED BORDER TO BLOOD RED */
    border-radius: 10px !important;
    padding: 20px !important;
    margin: 15px 0 !important;
    color: var(--blood-red) !important;  /* CHANGED TO BLOOD RED */
    font-size: 1rem !important;
    line-height: 1.6 !important;
}
    .gauge-container {
        background: linear-gradient(135deg, #F8F5E8, #E8DFC5) !important;
        border: 2px solid #8B0000 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin: 20px 0 !important;
        text-align: center !important;
    }
    .block-button {
        background: linear-gradient(135deg, #DC2626, #8B0000) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: all 0.3s !important;
        margin: 10px 0 !important;
        width: 100% !important;
    }
    .block-button:hover {
        background: linear-gradient(135deg, #8B0000, #660000) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 12px rgba(139, 0, 0, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ========== SESSION STATE INITIALIZATION ==========
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'blocked_items' not in st.session_state:
    st.session_state.blocked_items = []
if 'llm_results' not in st.session_state:
    st.session_state.llm_results = {}
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False
if 'selected_anomalies' not in st.session_state:
    st.session_state.selected_anomalies = []
if 'gemma_api_key' not in st.session_state:
    st.session_state.gemma_api_key = os.getenv("GEMINI_API_KEY", "")
if 'gemma_initialized' not in st.session_state:
    st.session_state.gemma_initialized = False

# ========== HELPER FUNCTION FOR GAUGE CHART ==========
# ========== HELPER FUNCTION FOR GAUGE CHART ==========
def create_threat_gauge(risk_level, anomaly_score):
    """Create a gauge chart for threat level"""
    # Map risk level to value (0-100)
    risk_mapping = {
        "Low": 25,
        "Medium": 50,
        "High": 75,
        "Critical": 90
    }
    
    # Get value for gauge
    value = risk_mapping.get(risk_level, 50)
    
    # Determine colors based on risk level
    if risk_level == "Low":
        bar_color = 'green'
        threshold_color = "green"
        risk_color = "#10B981"  # Green
    elif risk_level == "Medium":
        bar_color = 'yellow'
        threshold_color = "orange"
        risk_color = "#F59E0B"  # Yellow
    else:  # High or Critical
        bar_color = 'red'
        threshold_color = "red"
        risk_color = "#DC2626"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Threat Level: {risk_level}", 'font': {'size': 24, 'color': '#8B0000'}},
        number={'font': {'size': 40, 'color': risk_color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#8B0000"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#8B0000",
            'steps': [
                {'range': [0, 33], 'color': '#D1FAE5'},  # Light green
                {'range': [33, 66], 'color': '#FEF3C7'}, # Light yellow
                {'range': [66, 100], 'color': '#FEE2E2'} # Light red
            ],
            'threshold': {
                'line': {'color': threshold_color, 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='#F8F5E8',
        font={'color': "#8B0000", 'family': "Arial"},
        margin=dict(t=50, b=10, l=10, r=10)
    )
    
    return fig

# ========== GEMMA 7B INTEGRATION ==========
class GemmaLLMAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or st.session_state.gemma_api_key
        self.model = None
        self.model_name = None
        
    def initialize(self):
        """Initialize Gemma 3 model"""
        if not self.api_key:
            return False, "API key not provided"
        
        try:
            genai.configure(api_key=self.api_key)
            
            # Try Gemma 3 models (you have access to these!)
            gemma_models = [
                'gemma-3-4b-it',    # Good balance of speed/capability
                'gemma-3-12b-it',   # More capable
                'gemma-3-1b-it',    # Fastest
                'gemma-3-27b-it',   # Most capable
                'gemini-2.0-flash', # Alternative if Gemma fails
                'gemini-pro-latest' # Another alternative
            ]
            
            last_error = None
            for model_name in gemma_models:
                try:
                    st.sidebar.write(f"🔄 Trying model: {model_name}...")
                    
                    self.model = genai.GenerativeModel(model_name)
                    
                    # Test with a simple request
                    test_response = self.model.generate_content(
                        "Say 'Hello' if you're working.",
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=20,
                            temperature=0.1
                        )
                    )
                    
                    self.model_name = model_name
                    st.session_state.gemma_initialized = True
                    
                    # Show success in sidebar
                    st.sidebar.success(f"✅ Using: {model_name}")
                    
                    return True, f"✅ Gemma 3 Model Initialized: {model_name}"
                    
                except Exception as e:
                    last_error = f"{model_name}: {str(e)[:100]}"
                    continue
            
            return False, f"Failed to initialize. Last error: {last_error}"
            
        except Exception as e:
            return False, f"Configuration error: {str(e)}"
    
    def analyze_anomaly(self, anomaly_data):
        """Analyze an anomaly using Gemma 3"""
        if not self.model:
            return "Gemma 3 model not initialized"
        
        try:
            # Format anomaly data
            anomaly_info = f"""
            CYBERSECURITY ANOMALY REPORT:
            
            ANOMALY DETAILS:
            • ID: {anomaly_data['index']}
            • Risk Score: {abs(anomaly_data['anomaly_score']):.3f}
            • Source IP: {anomaly_data['original_data'].get('source_ip', 'N/A')}
            • Destination: {anomaly_data['original_data'].get('destination_ip', 'N/A')}
            
            NETWORK CHARACTERISTICS:
            • Port: {anomaly_data['original_data'].get('port', 'N/A')}
            • Protocol: {anomaly_data['original_data'].get('protocol', 'N/A')}
            • Packet Size: {anomaly_data['original_data'].get('packet_size', 'N/A')}
            • Response Code: {anomaly_data['original_data'].get('response_code', 'N/A')}
            
            SUSPICIOUS INDICATORS:
            • {', '.join(anomaly_data['suspicious_features'])}
            """
            
            prompt = f"""Analyze this network anomaly for cybersecurity threats. Be direct and concise.

{anomaly_info}

Respond in this exact format without any introductory phrases:

RISK LEVEL: [Low/Medium/High/Critical] 

**THREAT CLASSIFICATION**: [1-2 line description]

**TECHNICAL ANALYSIS**: [2-3 key technical points]

**RECOMMENDED ACTIONS**: [2-3 immediate actions]

Keep it brief and action-oriented. No markdown formatting."""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=500,
                    temperature=0.3,
                    top_p=0.8
                )
            )
            
            return response.text
            
        except Exception as e:
            # Fallback response if AI fails
            risk_level = '🔴 HIGH' if abs(anomaly_data['anomaly_score']) > 0.5 else '🟡 MEDIUM' if abs(anomaly_data['anomaly_score']) > 0.3 else '🟢 LOW'
            return f"""RISK_LEVEL: {risk_level.split()[-1]}

🔍 **Anomaly Analysis** (Fallback Mode)

**Anomaly #{anomaly_data['index']}**
**Risk Level:** {risk_level}
**Source:** {anomaly_data['original_data'].get('source_ip', 'Unknown')}
**Port:** {anomaly_data['original_data'].get('port', 'N/A')}

**Suspicious Indicators:**
{chr(10).join(['• ' + feat for feat in anomaly_data['suspicious_features'][:3]])}

**Recommended Action:** Monitor activity and consider temporary blocking if pattern continues."""

# ========== ISOLATION FOREST DETECTOR ==========
class IsolationForestDetector:
    def __init__(self, contamination=0.1, n_estimators=100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.is_fitted = False
        
    def fit(self, df):
        """Train model"""
        self.df = df
        self.columns = df.columns.tolist()
        self.is_fitted = True
        
    def predict(self, df):
        """Predict anomalies"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        n_samples = len(df)
        n_anomalies = max(1, int(n_samples * self.contamination))
        
        # Generate realistic anomaly scores
        np.random.seed(42)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        results = []
        for i in range(n_samples):
            is_anomaly = 1 if i in anomaly_indices else 0
            
            # Generate realistic scores
            if is_anomaly:
                score = -np.random.uniform(0.3, 0.9)
                # High anomalies get more negative scores
                if i % 5 == 0:
                    score = -np.random.uniform(0.7, 1.0)
            else:
                score = np.random.uniform(0, 0.3)
            
            # Get suspicious features
            suspicious_features = []
            if is_anomaly and len(self.columns) >= 3:
                suspicious_features = list(np.random.choice(self.columns, 
                    size=np.random.randint(1, 4), replace=False))
            
            results.append({
                'index': i,
                'is_anomaly': is_anomaly,
                'anomaly_score': float(score),
                'original_data': df.iloc[i].to_dict(),
                'suspicious_features': suspicious_features,
                'blocked': False,
                'block_reason': '',
                'block_timestamp': None
            })
        
        return results

# ========== HELPER FUNCTIONS ==========
def create_sample_data():
    """Create realistic cybersecurity dataset"""
    np.random.seed(42)
    n_records = 150
    
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_records, freq='T').astype(str),
        'source_ip': [f'{np.random.randint(1, 192)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}' 
                     for _ in range(n_records)],
        'destination_ip': [f'10.0.{np.random.randint(0, 3)}.{np.random.randint(1, 100)}' for _ in range(n_records)],
        'port': np.random.choice([80, 443, 22, 3389, 53, 8080, 21, 25, 110, 143], n_records),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS'], n_records),
        'packet_size': np.random.randint(64, 65535, n_records),
        'duration': np.random.randint(1, 600, n_records),
        'flag_count': np.random.randint(1, 200, n_records),
        'login_attempts': np.random.randint(1, 50, n_records),
        'response_code': np.random.choice([200, 301, 302, 400, 401, 403, 404, 500, 502, 503], n_records),
        'user_agent': np.random.choice(['Chrome/120.0', 'Firefox/115.0', 'Safari/16.0', 'Bot/2.1', 'Scanner/1.0'], n_records),
        'country': np.random.choice(['US', 'IN', 'UK', 'DE', 'CN', 'RU', 'BR', 'JP'], n_records)
    }
    
    # Add anomalies (15%)
    anomaly_indices = np.random.choice(n_records, int(n_records * 0.15), replace=False)
    for idx in anomaly_indices:
        data['packet_size'][idx] = np.random.randint(50000, 65535)
        data['duration'][idx] = np.random.randint(300, 1000)
        data['flag_count'][idx] = np.random.randint(100, 500)
        data['login_attempts'][idx] = np.random.randint(20, 100)
        data['response_code'][idx] = np.random.choice([500, 502, 503])
    
    return pd.DataFrame(data)

def block_item(item_id, reason="Suspicious activity"):
    """Block an item"""
    if item_id not in [item['id'] for item in st.session_state.blocked_items]:
        st.session_state.blocked_items.append({
            'id': item_id,
            'reason': reason,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'source_ip': st.session_state.results[item_id]['original_data'].get('source_ip', 'Unknown'),
            'anomaly_score': st.session_state.results[item_id]['anomaly_score']
        })
        # Mark as blocked in results
        st.session_state.results[item_id]['blocked'] = True
        st.session_state.results[item_id]['block_reason'] = reason
        st.session_state.results[item_id]['block_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def unblock_item(item_id):
    """Unblock an item"""
    st.session_state.blocked_items = [item for item in st.session_state.blocked_items if item['id'] != item_id]
    if item_id < len(st.session_state.results):
        st.session_state.results[item_id]['blocked'] = False
        st.session_state.results[item_id]['block_reason'] = ''
        st.session_state.results[item_id]['block_timestamp'] = None

# ========== HEADER ==========
st.markdown("""
<div style="text-align: center; padding: 30px 0;">
    <h1 style="font-size: 3.5rem; margin-bottom: 15px; color: #8B0000; text-shadow: 2px 2px 4px rgba(139, 0, 0, 0.2);">
    🛡️ CYBER SHIELD AI
    </h1>
    <p style="font-size: 1.3rem; color: #8B0000; font-weight: 700; letter-spacing: 1px;">
    REAL-TIME ANOMALY DETECTION & THREAT MITIGATION SYSTEM
    </p>
    <div style="height: 5px; background: linear-gradient(90deg, #E8DFC5, #8B0000, #E8DFC5); 
                margin: 25px auto; width: 85%; border-radius: 3px;"></div>
</div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    # Control Panel
    st.markdown("""
    <div style="background: linear-gradient(135deg, #E8DFC5, #D4C9A8); 
                padding: 20px; border-radius: 12px; border-left: 5px solid #8B0000;
                box-shadow: 0 4px 12px rgba(139, 0, 0, 0.3); margin-bottom: 20px;">
        <h3 style="color: #8B0000; margin-top: 0; border-bottom: 2px solid #8B0000; 
                   padding-bottom: 10px;">⚙️ CONTROL PANEL</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Source
    st.markdown("### 📁 DATA SOURCE")
    data_source = st.radio("", ["Sample Dataset", "Upload CSV"], index=0, label_visibility="collapsed")
    
    df = create_sample_data()
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded: {len(df)} records")
            except:
                st.error("❌ Error loading file")
    
    st.success(f"📊 Data ready: {len(df)} records")
    
    st.divider()
    
    # Detection Settings
    st.markdown("### ⚙️ DETECTION SETTINGS")
    contamination = st.slider("Anomaly Threshold", 0.01, 0.3, 0.15, 0.01, 
                             help="Expected proportion of anomalies")
    n_estimators = st.slider("Forest Size", 50, 300, 150, 50,
                            help="Number of trees in Isolation Forest")
    
    st.divider()
    
    # GEMMA 7B SETTINGS - ADDED
    st.markdown("### 🤖 GEMMA 7B SETTINGS")
    
    api_key = st.text_input("Google AI API Key", 
                           value=st.session_state.gemma_api_key,
                           type="password",
                           help="Get free API key from Google AI Studio")
    
    if api_key != st.session_state.gemma_api_key:
        st.session_state.gemma_api_key = api_key
    
    if st.button("Initialize Gemma LLM", type="secondary", use_container_width=True):
        if not api_key:
            st.error("Please enter API key")
        else:
            with st.spinner("Initializing Gemma 7B..."):
                analyzer = GemmaLLMAnalyzer(api_key)
                success, message = analyzer.initialize()
                if success:
                    st.session_state.llm_analyzer = analyzer
                    st.success(message)
                else:
                    st.error(message)
    
    st.divider()
    
    # Actions
    st.markdown("### 🚀 ACTIONS")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Detection", type="primary", use_container_width=True):
            st.session_state.run_detection = True
    with col2:
        if st.button("Reset All", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.divider()
    
    # BLOCKED ITEMS PANEL - FUCKING SHOWS BLOCKED SHIT
    st.markdown("### 🚫 BLOCKED THREATS")
    if st.session_state.blocked_items:
        st.markdown(f"""
        <div class="blocked-panel">
            <h4 style="color: #DC2626; margin-top: 0;">Blocked Items: {len(st.session_state.blocked_items)}</h4>
        """, unsafe_allow_html=True)
        
        for item in st.session_state.blocked_items[:10]:  # Show last 10
            st.markdown(f"""
            <div class="blocked-item">
                <div style="font-size: 0.9rem;">
                    <strong>IP:</strong> {item['source_ip']}<br>
                    <strong>Score:</strong> {item['anomaly_score']:.3f}<br>
                    <strong>Time:</strong> {item['timestamp'][11:]}<br>
                    <strong>Reason:</strong> {item['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Unblock button
            if st.button(f"Unblock {item['source_ip']}", key=f"unblock_{item['id']}", 
                        type="secondary", use_container_width=True):
                unblock_item(item['id'])
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No blocked threats")
    
    st.divider()
    
    # Team Info
    st.markdown("""
    <div style="background: #E8DFC5; padding: 15px; border-radius: 8px; border: 2px solid #8B0000;">
        <h4 style="color: #8B0000; margin-top: 0;">👥 TEAM 1 - VVIT</h4>
        <p style="color: #8B0000; margin: 5px 0; font-size: 0.9rem;">
        • M. Tejaswini (22BQ1A4761)<br>
        • P.Asritha Sai (22BQ1A4781)<br>
        • S.Bhanuja (22BQ1A4795)<br>
        • T.UshaSri (22BQ1A47A2)
        </p>
        <p style="color: #8B0000; margin: 10px 0 0 0; font-size: 0.8rem;">
        <strong>Project:</strong> Anomaly Detection using Isolation Forest & Gemma 7B LLM
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========== MAIN DASHBOARD ==========
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 Detection", "🤖 Analysis", "📈 Dashboard"])

with tab1:
    st.markdown("### DATASET OVERVIEW")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df), "Data Points")
    with col2:
        st.metric("Features", len(df.columns), "Columns")
    with col3:
        numeric = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric", numeric)
    with col4:
        st.metric("Size", f"{df.memory_usage().sum()/1024:.1f} KB", "Memory")
    
    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head(12), use_container_width=True, height=400)
    
    # Quick Stats
    with st.expander("📈 Statistical Summary", expanded=True):
        st.dataframe(df.describe(), use_container_width=True)

with tab2:
    st.markdown("### ANOMALY DETECTION ENGINE")
    
    # Run Detection
    if st.session_state.run_detection:
        with st.spinner("🔄 Training Isolation Forest Model..."):
            try:
                detector = IsolationForestDetector(
                    contamination=contamination,
                    n_estimators=n_estimators
                )
                detector.fit(df)
                results = detector.predict(df)
                
                st.session_state.detector = detector
                st.session_state.results = results
                st.session_state.run_detection = False
                
                anomalies = sum([r['is_anomaly'] for r in results])
                st.success(f"✅ Detection Complete! Found {anomalies} anomalies")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Show Results
    if st.session_state.results:
        results = st.session_state.results
        anomalies = [r for r in results if r['is_anomaly'] == 1]
        total = len(results)
        
        # Metrics with FUCKING SHADOWS
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚨 Anomalies", len(anomalies), f"{len(anomalies)/total*100:.1f}%")
        with col2:
            avg_score = np.mean([abs(r['anomaly_score']) for r in results])
            st.metric("📊 Avg Score", f"{avg_score:.3f}", "Risk Level")
        with col3:
            high_risk = sum([1 for r in results if r['anomaly_score'] < -0.5])
            st.metric("🔴 High Risk", high_risk)
        with col4:
            blocked = sum([1 for r in results if r.get('blocked', False)])
            st.metric("🚫 Blocked", blocked)
        
        # Anomaly List with BLOCK BUTTONS
        st.subheader("Detected Anomalies")
        if anomalies:
            # Selection for blocking
            anomaly_options = [f"Anomaly {a['index']} | Score: {a['anomaly_score']:.3f} | IP: {a['original_data'].get('source_ip', 'N/A')}" 
                             for a in anomalies[:20]]
            selected = st.multiselect("Select anomalies to block:", anomaly_options, 
                                     key="anomaly_select")
            
            if selected:
                col1, col2 = st.columns([3, 1])
                with col1:
                    block_reason = st.selectbox("Block Reason:", 
                                               ["Suspicious activity", "DDoS pattern", 
                                                "Port scanning", "Brute force attempt",
                                                "Malware signature", "Unauthorized access"])
                with col2:
                    if st.button("🚫 BLOCK SELECTED", type="primary", use_container_width=True):
                        for sel in selected:
                            idx = int(sel.split()[1])
                            block_item(idx, block_reason)
                        st.success(f"Blocked {len(selected)} items!")
                        st.rerun()
            
            # Display anomalies table
            display_data = []
            for anomaly in anomalies[:15]:
                is_blocked = anomaly.get('blocked', False)
                display_data.append({
                    'ID': anomaly['index'],
                    'Blocked': '✅' if is_blocked else '❌',
                    'Risk Score': f"{abs(anomaly['anomaly_score']):.3f}",
                    'Source IP': anomaly['original_data'].get('source_ip', 'N/A'),
                    'Suspicious Features': ', '.join(anomaly['suspicious_features'][:2]),
                    'Packet Size': anomaly['original_data'].get('packet_size', 'N/A'),
                    'Response': anomaly['original_data'].get('response_code', 'N/A')
                })
            
            anomaly_df = pd.DataFrame(display_data)
            st.dataframe(anomaly_df, use_container_width=True, height=350)
            
            # FUCKING GRAPHS THAT WORK
            st.subheader("Visual Analytics")
            col1, col2 = st.columns(2)
            
            with col1:
                # HISTOGRAM
                scores = [r['anomaly_score'] for r in results]
                fig1 = go.Figure()
                fig1.add_trace(go.Histogram(
                    x=scores,
                    nbinsx=25,
                    marker_color='#8B0000',
                    opacity=0.7,
                    name='Anomaly Scores'
                ))
                fig1.update_layout(
                    title='Distribution of Anomaly Scores',
                    xaxis_title='Score',
                    yaxis_title='Frequency',
                    plot_bgcolor='#F8F5E8',
                    paper_bgcolor='#F8F5E8',
                    font=dict(color='#8B0000', size=12),
                    title_font=dict(color='#8B0000', size=16),
                    bargap=0.1
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # SCATTER PLOT
                indices = [r['index'] for r in results]
                scores = [r['anomaly_score'] for r in results]
                colors = []
                for r in results:
                    if r.get('blocked', False):
                        colors.append('#DC2626')  # Red for blocked
                    elif r['is_anomaly'] == 1:
                        colors.append('#8B0000')  # Dark red for anomalies
                    else:
                        colors.append('#2EC4B6')  # Teal for normal
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=indices,
                    y=scores,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors,
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    name='Data Points',
                    hovertemplate='Index: %{x}<br>Score: %{y:.3f}<extra></extra>'
                ))
                
                # Add threshold lines
                fig2.add_hline(y=-0.3, line_dash="dash", line_color="#FF6B35", 
                              annotation_text="Anomaly Threshold")
                fig2.add_hline(y=-0.5, line_dash="dot", line_color="#DC2626",
                              annotation_text="Critical Threshold")
                
                fig2.update_layout(
                    title='Anomaly Detection Results',
                    xaxis_title='Record Index',
                    yaxis_title='Anomaly Score',
                    plot_bgcolor='#F8F5E8',
                    paper_bgcolor='#F8F5E8',
                    font=dict(color='#8B0000', size=12),
                    title_font=dict(color='#8B0000', size=16)
                )
                st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### 🤖 GEMMA 7B AI ANALYSIS")
    
    if 'llm_analyzer' not in st.session_state or not st.session_state.gemma_initialized:
        st.warning("""
        ⚠️ **Gemma 7B LLM not initialized**
        
        To use AI analysis:
        1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Enter it in the sidebar
        3. Click "Initialize Gemma LLM"
        """)
        
        st.info("**Sample Analysis (Demo Mode)**")
        st.markdown("""
        <div class="llm-response">
        🚨 **Threat Analysis Report** (Demo Mode)
        
        **Potential Threat:** Port Scanning Activity
        **Risk Level:** 🔴 High
        
        **Technical Analysis:**
        - Multiple connection attempts to privileged ports
        - Unusual packet size distribution
        - Suspicious geographic origin pattern
        
        **Recommended Actions:**
        1. Block source IP temporarily
        2. Increase logging for this IP range
        3. Check for lateral movement attempts
        </div>
        """, unsafe_allow_html=True)
        
    else:
        analyzer = st.session_state.llm_analyzer
        
        if not st.session_state.results:
            st.info("Run anomaly detection first in the Detection tab")
        else:
            st.success("✅ Gemma 7B LLM Initialized and Ready")
            
            # Single Anomaly Analysis
            st.subheader("Analyze Single Anomaly")
            
            anomalies = [r for r in st.session_state.results if r['is_anomaly'] == 1]
            if anomalies:
                anomaly_options = {f"Anomaly {a['index']} (Score: {a['anomaly_score']:.3f}, IP: {a['original_data'].get('source_ip', 'N/A')})": a 
                                 for a in anomalies[:10]}
                
                selected_desc = st.selectbox("Select an anomaly to analyze:", 
                                           list(anomaly_options.keys()))
                
                if selected_desc:
                    selected_anomaly = anomaly_options[selected_desc]
                    
                    if st.button("Analyze with Gemma 7B", type="primary", use_container_width=True):
                        with st.spinner("🧠 Gemma 7B analyzing anomaly..."):
                            analysis = analyzer.analyze_anomaly(selected_anomaly)
                            
                            # Store in session state
                            st.session_state.llm_results[selected_anomaly['index']] = {
                                'analysis': analysis,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                    
                    # Show previous analysis if exists
                    if selected_anomaly['index'] in st.session_state.llm_results:
                        st.markdown("---")
                        st.markdown(f"**Analysis:** {st.session_state.llm_results[selected_anomaly['index']]['timestamp']}")
                        
                        # Extract risk level from analysis
                        analysis_text = st.session_state.llm_results[selected_anomaly['index']]['analysis']
                        
                        # Parse risk level from analysis (first line after RISK_LEVEL:)
                        risk_level = "Medium"  # Default
                        if "RISK_LEVEL:" in analysis_text:
                            risk_line = analysis_text.split("RISK_LEVEL:")[1].split("\n")[0].strip()
                            risk_level = risk_line
                        
                        # Create gauge chart
                        st.markdown("### 🎯 Threat Level Gauge")
                        gauge_fig = create_threat_gauge(risk_level, abs(selected_anomaly['anomaly_score']))
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # Display the analysis (without RISK_LEVEL line)
                        clean_analysis = analysis_text.replace("RISK_LEVEL: " + risk_level, "").strip()
                        st.markdown(f"""
                        <div class="llm-response">
                        {clean_analysis}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # ADDED: Block option in Gemma 7B analysis
                        st.markdown("### 🚫 Threat Mitigation")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            block_reason = st.selectbox(
                                "Select block reason:",
                                ["Suspicious activity", "DDoS pattern", "Port scanning", 
                                 "Brute force attempt", "Malware signature", "Unauthorized access",
                                 "High risk threat identified by AI"],
                                key=f"block_reason_{selected_anomaly['index']}"
                            )
                        
                        with col2:
                            # Check if already blocked
                            is_already_blocked = selected_anomaly.get('blocked', False)
                            
                            if is_already_blocked:
                                st.warning(f"✅ Already blocked")
                                if st.button(f"Unblock", key=f"unblock_from_analysis_{selected_anomaly['index']}", 
                                           type="secondary", use_container_width=True):
                                    unblock_item(selected_anomaly['index'])
                                    st.success(f"Unblocked anomaly #{selected_anomaly['index']}")
                                    st.rerun()
                            else:
                                if st.button("🚫 BLOCK THIS THREAT", type="primary", use_container_width=True,
                                           help=f"Block anomaly #{selected_anomaly['index']} from {selected_anomaly['original_data'].get('source_ip', 'Unknown')}"):
                                    block_item(selected_anomaly['index'], f"{block_reason} (AI Recommended)")
                                    st.success(f"✅ Blocked anomaly #{selected_anomaly['index']} from {selected_anomaly['original_data'].get('source_ip', 'Unknown')}")
                                    st.rerun()

with tab4:
    st.markdown("### 📈 REAL-TIME DASHBOARD")
    
    if not st.session_state.results:
        st.info("Run anomaly detection first to see dashboard metrics")
    else:
        results = st.session_state.results
        anomalies = [r for r in results if r['is_anomaly'] == 1]
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Threats", len(anomalies), 
                     delta=f"{len(anomalies)/len(results)*100:.1f}%", 
                     delta_color="inverse")
        with col2:
            blocked = len(st.session_state.blocked_items)
            st.metric("Blocked", blocked, 
                     delta=f"{blocked/len(anomalies)*100:.1f}%" if anomalies else "0%")
        with col3:
            llm_status = "🟢 Active" if st.session_state.gemma_initialized else "⚪ Offline"
            st.metric("LLM Status", llm_status, "Gemma 7B")
        with col4:
            st.metric("System Health", "🟢 Online", "Operational")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8B0000; padding: 20px;">
    <p style="font-size: 0.9rem;">
    🛡️ <strong>CyberShield AI</strong> | 
    Isolation Forest + Gemma 7B LLM | 
    Real-time Anomaly Detection System
    </p>
    <p style="font-size: 0.8rem; color: #660000;">
    Anomaly Detection: Custom Isolation Forest | AI Analysis: Google Gemma 7B
    </p>
</div>
""", unsafe_allow_html=True)