import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Employee Retention Buddy",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.alert-card { border-left: 4px solid #ff4b4b; padding: 10px; margin: 5px 0;
              background: #2d1b1b; border-radius: 5px; }
.risk-high { color: #ff4b4b; font-weight: bold; }
.risk-medium { color: #ffa500; font-weight: bold; }
.risk-low { color: #00cc44; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(path, data=None, files=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, files=files, timeout=60)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


st.sidebar.image("https://img.icons8.com/fluency/96/handshake.png", width=70)
st.sidebar.title("Retention Buddy")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "👥 Employees", "💬 Check-in", "📈 Analytics", "⚙️ Settings"])

# ─── DASHBOARD ────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("HR Retention Dashboard")
    st.caption(f"Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    data = api_get("/dashboard")
    if not data:
        st.error("⚠️ Cannot connect to API on port 8000.")
        st.code("uvicorn backend.main:app --reload --port 8000")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", data.get("total_employees", 0))
    c2.metric("🔴 High Risk", data.get("high_risk_count", 0))
    c3.metric("🟡 Medium Risk", data.get("medium_risk_count", 0))
    c4.metric("🟢 Low Risk", data.get("low_risk_count", 0))

    st.divider()
    col_l, col_r = st.columns([2, 1])

    employees = data.get("employees", [])
    with col_l:
        st.subheader("Employee Risk Overview")
        if employees:
            df = pd.DataFrame(employees)
            color_map = {"High Risk": "#ff4b4b", "Medium Risk": "#ffa500",
                         "Low Risk": "#00cc44", "Unknown": "#888"}
            fig = px.bar(
                df.sort_values("risk_score", ascending=False).head(20),
                x="name", y="risk_score", color="risk_level",
                color_discrete_map=color_map,
                labels={"risk_score": "Risk Score (%)", "name": "Employee"},
                title="Top 20 by Risk Score"
            )
            fig.update_layout(xaxis_tickangle=-45, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No employee data yet.")

    with col_r:
        st.subheader("Risk Distribution")
        h, m, l = data.get("high_risk_count", 0), data.get("medium_risk_count", 0), data.get("low_risk_count", 0)
        if h + m + l > 0:
            fig_pie = go.Figure(go.Pie(
                labels=["High", "Medium", "Low"],
                values=[h, m, l],
                marker_colors=["#ff4b4b", "#ffa500", "#00cc44"],
                hole=0.4
            ))
            fig_pie.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🚨 Active Alerts")
    alerts = data.get("alerts", [])
    if alerts:
        for alert in alerts[:5]:
            emp_name = next((e["name"] for e in employees if e["id"] == alert.get("employee_id")), "Unknown")
            st.markdown(f"""
            <div class="alert-card">
                <strong>{alert.get('alert_type', '')}</strong> — {emp_name}<br>
                {alert.get('message', '')}<br>
                <small>Risk: {alert.get('risk_score', 0):.1f}% | {str(alert.get('created_at', ''))[:10]}</small>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ No active alerts")

    st.subheader("Recent Check-ins")
    convs = data.get("recent_conversations", [])
    if convs:
        st.dataframe(pd.DataFrame([{
            "Emp ID": c["employee_id"], "Week": c["week_number"],
            "Sentiment": c.get("sentiment", "N/A"), "Emotion": c.get("emotion", "N/A"),
            "Concern": c.get("problem_category", "none"), "Date": str(c["created_at"])[:10]
        } for c in convs]), use_container_width=True)


# ─── EMPLOYEES ────────────────────────────────────────────────────────────────
elif page == "👥 Employees":
    st.title("Employee Management")
    tab_view, tab_add = st.tabs(["View Employees", "Add Employee"])

    with tab_view:
        employees = api_get("/employees")
        if employees:
            df = pd.DataFrame(employees)
            search = st.text_input("🔍 Search by name or department")
            if search:
                mask = (df["name"].str.contains(search, case=False, na=False) |
                        df.get("department", pd.Series(dtype=str)).str.contains(search, case=False, na=False))
                df = df[mask]
            st.dataframe(df[["id", "employee_id", "name", "email", "department", "role"]], use_container_width=True)

            st.subheader("Employee Detail")
            emp_names = {e["name"]: e["id"] for e in employees}
            selected = st.selectbox("Select employee", list(emp_names.keys()))
            if selected:
                eid = emp_names[selected]
                summary = api_get(f"/employees/{eid}/trends")
                recs = api_get(f"/recommendations/{eid}")

                if summary:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Participation", f"{summary.get('participation_rate', 0)*100:.0f}%")
                    c2.metric("Avg Sentiment", f"{summary.get('avg_sentiment', 0):.2f}")
                    c3.metric("Primary Concern", summary.get("primary_concern", "none").replace("_", " ").title())

                    trend = summary.get("sentiment_trend", [])
                    if trend:
                        tdf = pd.DataFrame(trend)
                        fig = px.line(tdf, x="week", y="sentiment_score",
                                      title=f"Sentiment Trend — {selected}", markers=True)
                        fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                                      annotation_text="Neutral")
                        st.plotly_chart(fig, use_container_width=True)

                if recs and recs.get("recommendations"):
                    st.subheader("📋 Recommendations")
                    for rec in recs["recommendations"]:
                        with st.expander(f"[{rec['priority']}] {rec['title']}"):
                            st.write(rec["description"])
                            st.write(f"**Owner:** {rec['owner']}")
                            for item in rec["action_items"]:
                                st.write(f"  • {item}")
        else:
            st.info("No employees found.")

    with tab_add:
        st.subheader("Add New Employee")
        with st.form("add_emp"):
            c1, c2 = st.columns(2)
            emp_id = c1.text_input("Employee ID*")
            name = c2.text_input("Full Name*")
            email = c1.text_input("Email*")
            department = c2.text_input("Department")
            role = c1.text_input("Role")
            manager_id = c2.text_input("Manager ID")
            joining_date = st.date_input("Joining Date", datetime.today())
            if st.form_submit_button("Add Employee", type="primary"):
                if emp_id and name and email:
                    result = api_post("/employees", {
                        "employee_id": emp_id, "name": name, "email": email,
                        "department": department, "role": role, "manager_id": manager_id,
                        "joining_date": joining_date.isoformat() + "T00:00:00"
                    })
                    st.success(f"✅ Added {name}") if result else st.error("Failed to add employee.")
                else:
                    st.warning("Employee ID, Name and Email are required.")


# ─── CHECK-IN ─────────────────────────────────────────────────────────────────
elif page == "💬 Check-in":
    st.title("Employee Check-in")

    employees = api_get("/employees")
    if not employees:
        st.error("No employees found.")
        st.stop()

    emp_map = {f"{e['name']} ({e['employee_id']})": e["id"] for e in employees}
    selected_emp = st.selectbox("Employee", list(emp_map.keys()))
    emp_id = emp_map[selected_emp]
    week_number = st.number_input("Week Number", min_value=1, max_value=52, value=1)

    st.divider()
    tab_voice, tab_text = st.tabs(["🎤 Voice Check-in", "📝 Text Check-in"])

    def show_result(result):
        st.success("✅ Analysis Complete")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment", result.get("sentiment", {}).get("sentiment", "N/A").title())
        c2.metric("Score", f"{result.get('sentiment', {}).get('sentiment_score', 0):.2f}")
        c3.metric("Emotion", result.get("emotion", {}).get("emotion", "N/A").title())
        c4.metric("Risk", result.get("risk", {}).get("risk_level", "N/A"))

        if result.get("transcript"):
            st.subheader("📝 Transcript")
            st.write(result["transcript"])

        st.subheader("🤖 AI Response")
        st.info(result.get("ai_response", ""))

        voice = result.get("voice", {})
        if voice:
            vc1, vc2, vc3 = st.columns(3)
            vc1.metric("Voice Confidence", f"{voice.get('voice_confidence', 0):.2f}")
            vc2.metric("Stress Level", f"{voice.get('stress_level', 0):.2f}")
            vc3.metric("Speaking Speed", f"{voice.get('speaking_speed', 0):.1f}")

        topics = result.get("topics", [])
        if topics:
            st.write("**Topics:**", ", ".join(topics))

        score = result.get("risk", {}).get("risk_score", 0)
        if score >= 60:
            st.error(f"⚠️ HIGH RISK: {score:.1f}%")
        elif score >= 30:
            st.warning(f"⚠️ MEDIUM RISK: {score:.1f}%")
        else:
            st.success(f"✅ LOW RISK: {score:.1f}%")

    with tab_voice:
        st.info("Upload a WAV/MP3 recording of the employee's response.")
        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
        if audio_file and st.button("Process Voice Check-in", type="primary"):
            with st.spinner("Transcribing → Analyzing → Generating response..."):
                files = {"audio_file": (audio_file.name, audio_file.read(), audio_file.type)}
                result = api_post(
                    f"/process-voice?employee_id={emp_id}&week_number={week_number}",
                    files=files
                )
            if result:
                show_result(result)
            else:
                st.error("Processing failed. Check API logs.")

    with tab_text:
        text_input = st.text_area("Employee's response", height=150,
                                   placeholder="Type or paste the employee's response here...")
        if st.button("Analyze", type="primary") and text_input:
            with st.spinner("Analyzing..."):
                result = api_post("/analyze-text", {
                    "employee_id": emp_id, "week_number": week_number, "text": text_input
                })
            if result:
                show_result(result)


# ─── ANALYTICS ────────────────────────────────────────────────────────────────
elif page == "📈 Analytics":
    st.title("Retention Analytics")

    dashboard = api_get("/dashboard")
    employees = api_get("/employees")

    if not employees:
        st.info("No data available yet. Run setup first.")
        st.stop()

    st.subheader("Department Risk Overview")
    if dashboard:
        emp_data = dashboard.get("employees", [])
        if emp_data:
            df = pd.DataFrame(emp_data)
            if "department" in df.columns:
                dept_df = df.groupby("department")["risk_score"].mean().reset_index()
                dept_df.columns = ["Department", "Avg Risk Score"]
                fig = px.bar(dept_df.sort_values("Avg Risk Score", ascending=False),
                             x="Department", y="Avg Risk Score",
                             color="Avg Risk Score", color_continuous_scale="RdYlGn_r",
                             title="Avg Risk Score by Department")
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Employee Trend Analysis")
    emp_map = {f"{e['name']} ({e['employee_id']})": e["id"] for e in employees}
    selected = st.selectbox("Select Employee", list(emp_map.keys()))
    if selected:
        eid = emp_map[selected]
        summary = api_get(f"/employees/{eid}/trends")
        if summary and summary.get("sentiment_trend"):
            tdf = pd.DataFrame(summary["sentiment_trend"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tdf["week"], y=tdf["sentiment_score"],
                                     name="Sentiment", mode="lines+markers",
                                     line=dict(color="#00cc44")))
            if "risk_score" in tdf.columns:
                fig.add_trace(go.Scatter(x=tdf["week"], y=tdf["risk_score"] / 100,
                                         name="Risk (normalized)", mode="lines+markers",
                                         line=dict(color="#ff4b4b", dash="dash")))
            fig.update_layout(title=f"Trends — {selected}",
                               xaxis_title="Week", yaxis_title="Score", height=350)
            st.plotly_chart(fig, use_container_width=True)

            topics = summary.get("recurring_topics", {})
            if topics:
                st.write("**Recurring Topics:**")
                st.bar_chart(pd.DataFrame(list(topics.items()), columns=["Topic", "Count"]).set_index("Topic"))
        else:
            st.info("No trend data available for this employee yet.")


# ─── SETTINGS ─────────────────────────────────────────────────────────────────
elif page == "⚙️ Settings":
    st.title("Settings & Configuration")

    health = api_get("/health")
    if health:
        st.success(f"✅ API Status: {health.get('status', 'unknown').upper()}")
    else:
        st.error("❌ API Unreachable — start with: `uvicorn backend.main:app --reload`")

    st.subheader("TTS Engine")
    st.info("""
**Available engines (set in `backend/config.py` or env vars):**
- `vibevoice` — Microsoft VibeVoice 1.5B (local, high quality)
- `elevenlabs` — ElevenLabs API (cloud, ultra-realistic) — set `ELEVENLABS_API_KEY`
- `coqui` — Coqui TTS (local, lightweight fallback)
    """)

    st.subheader("Generate Synthetic Training Data")
    n = st.slider("Number of synthetic employees", 10, 200, 50)
    if st.button("Generate & Seed Database"):
        import subprocess, sys
        with st.spinner("Generating..."):
            r = subprocess.run(
                [sys.executable, "-m", "data.generate_synthetic", "--employees", str(n)],
                capture_output=True, text=True
            )
        if r.returncode == 0:
            st.success(r.stdout)
        else:
            st.error(r.stderr)

    st.subheader("API Reference")
    for ep, desc in {
        "POST /employees": "Create employee",
        "GET /employees": "List employees",
        "GET /employees/{id}/trends": "Employee trends",
        "POST /process-voice": "Full voice check-in pipeline (Whisper STT → NLP → Risk → VibeVoice/ElevenLabs TTS)",
        "POST /analyze-text": "Text-based check-in",
        "GET /dashboard": "HR dashboard data",
        "GET /recommendations/{id}": "Retention recommendations",
        "POST /alerts/{id}/resolve": "Resolve alert",
    }.items():
        st.write(f"`{ep}` — {desc}")
