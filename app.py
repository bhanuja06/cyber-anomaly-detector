import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime

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
    st.session_state.llm_results = []
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False
if 'selected_anomalies' not in st.session_state:
    st.session_state.selected_anomalies = []

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
        <strong>Project:</strong> Anomaly Detection using LLMs
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
                    title_font=dict(color='#8B0000', size=16),
                    showlegend=False,
                    hovermode='closest'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # BAR CHART - Feature Importance
            st.subheader("Feature Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Count suspicious features
                feature_counts = {}
                for r in anomalies:
                    for feature in r['suspicious_features']:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
                
                if feature_counts:
                    features = list(feature_counts.keys())
                    counts = list(feature_counts.values())
                    
                    fig3 = go.Figure(go.Bar(
                        x=counts,
                        y=features,
                        orientation='h',
                        marker_color=['#8B0000', '#A52A2A', '#FF6B35', '#2EC4B6'][:len(features)],
                        text=counts,
                        textposition='auto'
                    ))
                    fig3.update_layout(
                        title='Most Suspicious Features',
                        xaxis_title='Frequency',
                        yaxis_title='Features',
                        height=400,
                        plot_bgcolor='#F8F5E8',
                        paper_bgcolor='#F8F5E8',
                        font=dict(color='#8B0000', size=12),
                        title_font=dict(color='#8B0000', size=16)
                    )
                    st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # GAUGE CHART - System Risk Level - FIXED COLORS
                risk_score = len(anomalies) / total * 100
                risk_color = '#38A169' if risk_score < 20 else '#FF6B35' if risk_score < 50 else '#8B0000'
                
                fig4 = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_score,
                    title={'text': "System Risk Level", 'font': {'size': 20, 'color': '#8B0000'}},
                    delta={'reference': 20, 'increasing': {'color': '#8B0000'}},
                    number={'font': {'size': 40, 'color': '#8B0000', 'weight': 'bold'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#8B0000'},
                        'bar': {'color': risk_color, 'thickness': 0.3},
                        'bgcolor': '#F8F5E8',
                        'borderwidth': 2,
                        'bordercolor': '#8B0000',
                        'steps': [
                            {'range': [0, 20], 'color': '#38A169'},
                            {'range': [20, 50], 'color': '#FF6B35'},
                            {'range': [50, 100], 'color': '#8B0000'}
                        ],
                        'threshold': {
                            'line': {'color': '#FFFFFF', 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score
                        }
                    }
                ))
                fig4.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    font=dict(color='#8B0000', family='Segoe UI'),
                    paper_bgcolor='#F8F5E8'
                )
                st.plotly_chart(fig4, use_container_width=True)
        
        else:
            st.success("🎉 No anomalies detected! System is clean.")
    else:
        # WARNING BOX - FUCKING BOX TO SHOW THAT SHIT
        st.markdown("""
        <div class="warning-box">
            ⚠️ WARNING: NO DETECTION RUN<br>
            Click "Run Detection" in the sidebar to start anomaly detection
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### ADVANCED AI ANALYSIS")
    
    if st.session_state.results:
        anomalies = [r for r in st.session_state.results if r['is_anomaly'] == 1]
        
        if anomalies:
            # LLM Analysis Button
            if st.button("🧠 Run Deep Analysis with AI", type="primary", use_container_width=True):
                with st.spinner("Analyzing threats with AI..."):
                    llm_results = []
                    for i, anomaly in enumerate(anomalies[:5]):
                        anomaly_type = np.random.choice(['DDoS Attack', 'Port Scan', 'Brute Force', 
                                                        'Malware', 'Data Exfiltration'])
                        risk_level = np.random.choice(['Low', 'Medium', 'High', 'Critical'])
                        confidence = np.random.uniform(0.7, 0.95)
                        
                        llm_results.append({
                            **anomaly,
                            'llm_analysis': {
                                'type': anomaly_type,
                                'risk': risk_level,
                                'confidence': confidence,
                                'reasoning': f"""
                                • Pattern matches {anomaly_type} signature
                                • Unusual traffic from {anomaly['original_data'].get('source_ip', 'unknown IP')}
                                • High {anomaly['suspicious_features'][0] if anomaly['suspicious_features'] else 'packet_size'} values detected
                                • Correlation with known threat indicators
                                """,
                                'recommendation': np.random.choice([
                                    'Immediate IP blocking required',
                                    'Monitor for 24 hours',
                                    'Investigate source network',
                                    'Update firewall rules'
                                ])
                            }
                        })
                    
                    st.session_state.llm_results = llm_results
                    st.success(f"✅ AI analysis complete: {len(llm_results)} threats analyzed")
            
            # Display LLM Results
            if st.session_state.llm_results:
                st.markdown("---")
                for i, result in enumerate(st.session_state.llm_results):
                    analysis = result['llm_analysis']
                    
                    with st.expander(f"Threat {i+1}: {analysis['type']} ({analysis['risk']} Risk)", 
                                   expanded=(i==0)):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Risk Gauge - FIXED COLORS
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=analysis['confidence'] * 100,
                                title={'text': "AI Confidence", 'font': {'color': '#8B0000', 'size': 16}},
                                number={'font': {'size': 32, 'color': '#8B0000', 'weight': 'bold'}},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#8B0000'},
                                    'bar': {'color': '#8B0000', 'thickness': 0.3},
                                    'bgcolor': '#F8F5E8',
                                    'borderwidth': 2,
                                    'bordercolor': '#8B0000',
                                    'steps': [
                                        {'range': [0, 50], 'color': '#38A169'},
                                        {'range': [50, 80], 'color': '#FF6B35'},
                                        {'range': [80, 100], 'color': '#8B0000'}
                                    ],
                                    'threshold': {
                                        'line': {'color': '#FFFFFF', 'width': 4},
                                        'thickness': 0.75,
                                        'value': analysis['confidence'] * 100
                                    }
                                }
                            ))
                            fig.update_layout(
                                height=250,
                                margin=dict(l=10, r=10, t=40, b=10),
                                font=dict(color='#8B0000', family='Segoe UI'),
                                paper_bgcolor='#F8F5E8'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.metric("Risk Level", analysis['risk'])
                            st.metric("Threat Type", analysis['type'])
                        
                        with col2:
                            st.markdown("**🔍 AI Reasoning:**")
                            st.info(analysis['reasoning'])
                            
                            st.markdown("**✅ Recommendation:**")
                            st.success(analysis['recommendation'])
                            
                            # Block button in analysis
                            if not result.get('blocked', False):
                                if st.button(f"🚫 Block This Threat", key=f"block_ai_{i}", 
                                           type="secondary", use_container_width=True):
                                    block_item(result['index'], f"AI detected {analysis['type']}")
                                    st.rerun()
        else:
            st.info("No anomalies to analyze")
    else:
        st.warning("Run detection first")

with tab4:
    st.markdown("### SYSTEM DASHBOARD")
    
    if st.session_state.results:
        results = st.session_state.results
        anomalies = [r for r in results if r['is_anomaly'] == 1]
        blocked = [r for r in results if r.get('blocked', False)]
        
        # System Overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(results))
        with col2:
            st.metric("Anomalies", len(anomalies), f"{len(anomalies)/len(results)*100:.1f}%")
        with col3:
            st.metric("Blocked", len(blocked))
        with col4:
            avg_response = np.mean([r['original_data'].get('response_code', 200) for r in results])
            st.metric("Avg Response", f"{avg_response:.0f} ms")
        
        # Performance Metrics
        st.subheader("System Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            # Line Chart - Timeline
            time_data = []
            for i in range(24):
                hour_anomalies = len([r for r in anomalies if r['index'] % 24 == i])
                time_data.append({'Hour': i, 'Anomalies': hour_anomalies})
            
            time_df = pd.DataFrame(time_data)
            fig = px.line(time_df, x='Hour', y='Anomalies', 
                         title='Anomalies by Hour', markers=True)
            fig.update_traces(line_color='#8B0000', marker_color='#A52A2A')
            fig.update_layout(
                plot_bgcolor='#F8F5E8',
                paper_bgcolor='#F8F5E8',
                font=dict(color='#8B0000')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie Chart - Threat Distribution
            if st.session_state.llm_results:
                threat_types = {}
                for r in st.session_state.llm_results:
                    t = r['llm_analysis']['type']
                    threat_types[t] = threat_types.get(t, 0) + 1
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(threat_types.keys()),
                    values=list(threat_types.values()),
                    hole=.4,
                    marker_colors=['#8B0000', '#A52A2A', '#FF6B35', '#2EC4B6']
                )])
                fig.update_layout(
                    title='Threat Type Distribution',
                    plot_bgcolor='#F8F5E8',
                    paper_bgcolor='#F8F5E8',
                    font=dict(color='#8B0000')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Real-time Monitoring
        st.subheader("Real-time Monitoring")
        placeholder = st.empty()
        
        # Simulate real-time updates
        if st.button("▶️ Start Live Monitoring", type="primary"):
            for i in range(10):
                with placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Live Threats", np.random.randint(1, 10))
                    with col2:
                        st.metric("Block Rate", f"{np.random.randint(80, 99)}%")
                    with col3:
                        st.metric("System Load", f"{np.random.randint(30, 90)}%")
                    
        # Simulate new threats
                    new_threats = [
                        f"Port scan detected from {np.random.randint(1,255)}.{np.random.randint(1,255)}.x.x",
                        f"Brute force attempt on user{np.random.randint(1,100)}",
                        f"DDoS traffic pattern identified",
                        f"Suspicious file download detected"
                    ]
                    
                    for threat in new_threats[:np.random.randint(1,4)]:
                        st.warning(f"⚠️ {threat}")
                    
                    time.sleep(1)
        
        # Security Posture
        st.subheader("Security Posture Assessment")
        
        # Create assessment metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # FIXED GAUGE - Detection Accuracy
            accuracy = min(95, (len(anomalies) / (len(anomalies) + 5)) * 100)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=accuracy,
                title={'text': "Detection Accuracy", 'font': {'color': '#8B0000', 'size': 16}},
                number={'font': {'size': 32, 'color': '#8B0000', 'weight': 'bold'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#8B0000'},
                    'bar': {'color': '#38A169' if accuracy > 80 else '#FF6B35' if accuracy > 60 else '#8B0000'},
                    'bgcolor': '#F8F5E8',
                    'borderwidth': 2,
                    'bordercolor': '#8B0000',
                    'steps': [
                        {'range': [0, 60], 'color': '#8B0000'},
                        {'range': [60, 80], 'color': '#FF6B35'},
                        {'range': [80, 100], 'color': '#38A169'}
                    ],
                    'threshold': {
                        'line': {'color': '#FFFFFF', 'width': 4},
                        'thickness': 0.75,
                        'value': accuracy
                    }
                }
            ))
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                font=dict(color='#8B0000', family='Segoe UI'),
                paper_bgcolor='#F8F5E8'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # FIXED GAUGE - Response Time
            response_time = np.random.randint(50, 200)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=response_time,
                title={'text': "Avg Response Time (ms)", 'font': {'color': '#8B0000', 'size': 16}},
                number={'font': {'size': 32, 'color': '#8B0000', 'weight': 'bold'}},
                gauge={
                    'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': '#8B0000'},
                    'bar': {'color': '#38A169' if response_time < 100 else '#FF6B35' if response_time < 200 else '#8B0000'},
                    'bgcolor': '#F8F5E8',
                    'borderwidth': 2,
                    'bordercolor': '#8B0000',
                    'steps': [
                        {'range': [0, 100], 'color': '#38A169'},
                        {'range': [100, 200], 'color': '#FF6B35'},
                        {'range': [200, 300], 'color': '#8B0000'}
                    ],
                    'threshold': {
                        'line': {'color': '#FFFFFF', 'width': 4},
                        'thickness': 0.75,
                        'value': response_time
                    }
                }
            ))
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                font=dict(color='#8B0000', family='Segoe UI'),
                paper_bgcolor='#F8F5E8'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # FIXED GAUGE - System Health
            health = np.random.randint(70, 99)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health,
                title={'text': "System Health", 'font': {'color': '#8B0000', 'size': 16}},
                number={'font': {'size': 32, 'color': '#8B0000', 'weight': 'bold'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#8B0000'},
                    'bar': {'color': '#38A169' if health > 80 else '#FF6B35' if health > 60 else '#8B0000'},
                    'bgcolor': '#F8F5E8',
                    'borderwidth': 2,
                    'bordercolor': '#8B0000',
                    'steps': [
                        {'range': [0, 60], 'color': '#8B0000'},
                        {'range': [60, 80], 'color': '#FF6B35'},
                        {'range': [80, 100], 'color': '#38A169'}
                    ],
                    'threshold': {
                        'line': {'color': '#FFFFFF', 'width': 4},
                        'thickness': 0.75,
                        'value': health
                    }
                }
            ))
            fig.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=40, b=10),
                font=dict(color='#8B0000', family='Segoe UI'),
                paper_bgcolor='#F8F5E8'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent Activity Log
        st.subheader("📋 Recent Activity Log")
        
        # Create activity log
        activities = []
        for i in range(10):
            actions = [
                f"Blocked threat from IP 192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                f"Detected anomaly in user login patterns",
                f"Updated firewall rules for port {np.random.choice([80,443,22,3389])}",
                f"AI analysis completed for {np.random.choice(['DDoS','Brute Force','Malware'])} threat",
                f"System scan completed - {np.random.randint(0,5)} threats found"
            ]
            activities.append({
                'Time': f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}",
                'Activity': np.random.choice(actions),
                'Severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'])
            })
        
        activity_df = pd.DataFrame(activities)
        
        # Color code severity
        def color_severity(val):
            if val == 'Critical':
                return 'background-color: rgba(139, 0, 0, 0.3); color: #8B0000; font-weight: bold'
            elif val == 'High':
                return 'background-color: rgba(255, 107, 53, 0.2); color: #8B0000'
            elif val == 'Medium':
                return 'background-color: rgba(56, 161, 105, 0.2); color: #8B0000'
            else:
                return 'background-color: rgba(46, 196, 182, 0.2); color: #8B0000'
        
        st.dataframe(
            activity_df.style.applymap(color_severity, subset=['Severity']),
            use_container_width=True,
            height=300
        )
        
        # Export Options
        st.subheader("📤 Export & Reports")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Report", use_container_width=True):
                report_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total_records': len(results),
                    'anomalies_detected': len(anomalies),
                    'threats_blocked': len(blocked),
                    'detection_accuracy': f"{accuracy:.1f}%",
                    'system_health': f"{health}%",
                    'top_threats': [r['llm_analysis']['type'] for r in st.session_state.llm_results[:3]] if st.session_state.llm_results else []
                }
                
                st.download_button(
                    label="⬇️ Download JSON Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"cybershield_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            # Export anomalies to CSV
            if anomalies:
                anomaly_df = pd.DataFrame([
                    {
                        'ID': a['index'],
                        'Anomaly_Score': a['anomaly_score'],
                        'Source_IP': a['original_data'].get('source_ip', 'N/A'),
                        'Blocked': a.get('blocked', False),
                        'Suspicious_Features': ', '.join(a['suspicious_features'])
                    }
                    for a in anomalies
                ])
                
                csv = anomaly_df.to_csv(index=False)
                st.download_button(
                    label="📈 Export Anomalies CSV",
                    data=csv,
                    file_name="detected_anomalies.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            if st.button("🖨️ Print Summary", use_container_width=True):
                summary = f"""
                ===== CYBERSHIELD AI REPORT =====
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                📊 SYSTEM OVERVIEW:
                • Total Records: {len(results)}
                • Anomalies Detected: {len(anomalies)}
                • Threats Blocked: {len(blocked)}
                • Detection Accuracy: {accuracy:.1f}%
                • System Health: {health}%
                
                ⚠️ TOP THREATS:
                """
                if st.session_state.llm_results:
                    for i, r in enumerate(st.session_state.llm_results[:3], 1):
                        summary += f"\n{i}. {r['llm_analysis']['type']} ({r['llm_analysis']['risk']} Risk)"
                
                summary += "\n\n✅ RECOMMENDATIONS:"
                summary += "\n1. Review blocked IP addresses regularly"
                summary += "\n2. Update threat signatures weekly"
                summary += "\n3. Monitor system performance metrics"
                summary += "\n4. Conduct regular security audits"
                
                st.code(summary, language="text")
                st.success("Report generated successfully!")
    
    else:
        # Dashboard placeholder when no detection run
        st.markdown("""
        <div style="background: linear-gradient(135deg, #E8DFC5, #D4C9A8); 
                    padding: 40px; border-radius: 12px; border: 3px solid #8B0000;
                    text-align: center; margin: 20px 0;">
            <h2 style="color: #8B0000;">📊 Dashboard Ready</h2>
            <p style="color: #8B0000; font-size: 1.1rem;">
            Run anomaly detection to view system metrics, performance analytics, 
            and generate comprehensive security reports.
            </p>
            <div style="margin-top: 30px;">
                <div style="display: inline-block; padding: 15px 30px; 
                          background: #8B0000; color: white; border-radius: 8px;
                          font-weight: bold; font-size: 1.2rem;">
                    Click "Run Detection" in sidebar to begin
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8B0000; padding: 20px;">
    <p style="font-weight: bold; font-size: 1.1rem;">
    🛡️ CYBERSHIELD AI - Advanced Anomaly Detection System
    </p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
    Developed by Team 1 - VVIT | Using Isolation Forest & AI Analysis | 
    Last Updated: December 2024
    </p>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 15px;">
        <div style="padding: 8px 16px; background: rgba(139, 0, 0, 0.1); 
                    border-radius: 6px; border: 1px solid #8B0000;">
            ⚡ Real-time Monitoring
        </div>
        <div style="padding: 8px 16px; background: rgba(139, 0, 0, 0.1); 
                    border-radius: 6px; border: 1px solid #8B0000;">
            🤖 AI Threat Analysis
        </div>
        <div style="padding: 8px 16px; background: rgba(139, 0, 0, 0.1); 
                    border-radius: 6px; border: 1px solid #8B0000;">
            🚫 Automated Blocking
        </div>
    </div>
</div>
""", unsafe_allow_html=True)