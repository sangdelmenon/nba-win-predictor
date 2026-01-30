"""
NBA Game Predictor - Streamlit Dashboard with Injury Tracking
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from src.agent import NBAAgent
from src.data_collector import NBADataCollector
from src.player_impact import STAR_PLAYERS, get_injury_tracker


def display_prediction(result, team1, team2):
    """Display prediction results"""

    winner_color = "#4ade80" if result['win_probability'] > 0.65 else "#fbbf24" if result['win_probability'] > 0.55 else "#f87171"

    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 1rem; margin: 1rem 0;">
        <h2 style="color: #888; margin-bottom: 0.5rem;">Predicted Winner</h2>
        <h1 style="color: {winner_color}; font-size: 2.5rem; margin: 0;">{result['predicted_winner']}</h1>
        <p style="color: #888; font-size: 1.2rem; margin-top: 0.5rem;">
            Win Probability: <span style="color: #60a5fa; font-weight: bold;">{result['win_probability']*100:.0f}%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Injury alerts
    if result.get('team1_injuries') or result.get('team2_injuries'):
        st.markdown("### ⚠️ Injury Report")

        col1, col2 = st.columns(2)
        with col1:
            if result.get('team1_injuries'):
                st.markdown(f"**{team1}** ({result['team1_strength']:.0f}% strength)")
                for player, impact, status in result['team1_injuries']:
                    st.markdown(f"- 🚑 {player} - **{status}** (Impact: {impact:.0f})")
            else:
                st.markdown(f"**{team1}**: ✅ Fully healthy")

        with col2:
            if result.get('team2_injuries'):
                st.markdown(f"**{team2}** ({result['team2_strength']:.0f}% strength)")
                for player, impact, status in result['team2_injuries']:
                    st.markdown(f"- 🚑 {player} - **{status}** (Impact: {impact:.0f})")
            else:
                st.markdown(f"**{team2}**: ✅ Fully healthy")

    # Win probability gauges
    st.markdown("### 📊 Win Probability")
    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['team1_win_prob'] * 100,
            title={'text': f"🏠 {team1}", 'font': {'size': 16}},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4ade80" if result['team1_win_prob'] > 0.5 else "#60a5fa"},
                'steps': [
                    {'range': [0, 50], 'color': "#1e293b"},
                    {'range': [50, 100], 'color': "#0f172a"}
                ],
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,  width='stretch')

    with prob_col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['team2_win_prob'] * 100,
            title={'text': f"✈️ {team2}", 'font': {'size': 16}},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4ade80" if result['team2_win_prob'] > 0.5 else "#60a5fa"},
                'steps': [
                    {'range': [0, 50], 'color': "#1e293b"},
                    {'range': [50, 100], 'color': "#0f172a"}
                ],
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig,  width='stretch')

    # Team stats comparison
    stats1 = result.get('team1_stats', {})
    stats2 = result.get('team2_stats', {})

    if stats1 and stats2:
        st.markdown("### 📈 Team Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(f"{team1} Win %", f"{stats1.get('win_pct', 0)*100:.0f}%")
            st.metric(f"{team1} PPG", f"{stats1.get('avg_pts', 0):.1f}")
            st.metric(f"{team1} Last 10", f"{stats1.get('last_10_wins', 0)}-{10-stats1.get('last_10_wins', 0)}")

        with col2:
            st.metric(f"{team2} Win %", f"{stats2.get('win_pct', 0)*100:.0f}%")
            st.metric(f"{team2} PPG", f"{stats2.get('avg_pts', 0):.1f}")
            st.metric(f"{team2} Last 10", f"{stats2.get('last_10_wins', 0)}-{10-stats2.get('last_10_wins', 0)}")

    # AI Explanation
    st.markdown("### 🤖 AI Analysis")
    st.markdown(result.get('explanation', 'Analysis not available'))


# Page config
st.set_page_config(
    page_title="NBA Game Predictor",
    page_icon="🏀",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'collector' not in st.session_state:
    st.session_state.collector = NBADataCollector()
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = datetime.now().date()

# Header
st.markdown('<h1 style="text-align: center; color: #f9a825;">🏀 NBA Game Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #888;">AI-powered predictions with injury tracking</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Controls")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🤖 Initialize", width='stretch', key="btn_initialize"):
            with st.spinner("Loading model..."):
                st.session_state.agent = NBAAgent()
                st.session_state.agent.initialize(force_retrain=False)
            st.success("Ready!")

    with col2:
        if st.button("🔄 Retrain",  width='stretch', key="btn_retrain"):
            # Delete old model
            if os.path.exists('models/nba_predictor.joblib'):
                os.remove('models/nba_predictor.joblib')

            with st.spinner("Retraining models..."):
                st.session_state.agent = NBAAgent()
                st.session_state.agent.initialize(force_retrain=True)
            st.success("Retrained!")

    if st.session_state.agent and st.session_state.agent.is_ready:
        st.success("✅ Agent is ready")

        if st.session_state.agent.model_scores:
            st.subheader("Model Accuracy")
            for model, score in st.session_state.agent.model_scores.items():
                icon = "🏆" if model == st.session_state.agent.best_model_name else "📊"
                st.write(f"{icon} {model}: {score:.1%}")

    st.divider()

    # Roster Update
    st.header("📋 Roster Management")

    if st.button("🔄 Update Rosters",  width='stretch', key="btn_update_rosters"):
        with st.spinner("Fetching latest rosters from NBA API..."):
            try:
                from src.roster_updater import update_rosters
                update_rosters()
                st.success("✅ Rosters updated!")
                st.info("💡 Restart the app to load new rosters")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Injury Management
    st.header("🚑 Injury Management")

    # Auto-update button at the top
    if st.button("🔄 Auto-Update from Basketball Ref",  width='stretch', type="primary", key="btn_auto_update_injuries"):
        with st.spinner("Fetching live injury data..."):
            try:
                if st.session_state.agent:
                    count = st.session_state.agent.refresh_injuries_from_espn()
                    st.success(f"✅ Updated {count} injuries!")
                    st.rerun()
                else:
                    st.warning("Please initialize the agent first")
            except Exception as e:
                st.error(f"Error fetching injuries: {e}")

    st.divider()

    # Manual injury management
    if st.session_state.agent:
        teams_list = sorted(set(p['team'] for p in STAR_PLAYERS.values()))
        selected_team = st.selectbox("Select Team", teams_list, key="select_injury_team")

        team_players = {name: data for name, data in STAR_PLAYERS.items()
                        if data['team'] == selected_team}

        if team_players:
            # Sort by impact
            sorted_players = sorted(team_players.keys(),
                                    key=lambda x: team_players[x]['impact'],
                                    reverse=True)
            selected_player = st.selectbox("Select Player", sorted_players, key="select_injury_player")

            injury_status = st.selectbox("Status", ["Available", "Probable", "Questionable", "Out"], key="select_injury_status")
            injury_reason = st.text_input("Reason (optional)", key="input_injury_reason")

            if st.button("Update Status",  width='stretch', key="btn_update_injury_status"):
                if injury_status == "Available":
                    st.session_state.agent.clear_injury(selected_player)
                    st.success(f"✅ {selected_player} is available")
                else:
                    st.session_state.agent.set_injury(selected_player, injury_status, injury_reason)
                    st.success(f"🚑 {selected_player}: {injury_status}")

        # Show current injuries
        st.subheader("Current Injuries")
        tracker = get_injury_tracker()
        if tracker.injuries:
            for player, info in list(tracker.injuries.items())[:10]:
                st.write(f"🚑 {player}: {info['status']}")
        else:
            st.write("No injuries reported")

# Main tabs
tab1, tab2 = st.tabs(["📅 Scheduled Games", "🎯 Custom Matchup"])

# Tab 1: Scheduled Games
with tab1:
    st.subheader("Select a Date")

    date_col1, date_col2, date_col3, date_col4 = st.columns(4)

    with date_col1:
        if st.button("📆 Today",  width='stretch', key="today"):
            st.session_state.selected_date = datetime.now().date()
            st.rerun()

    with date_col2:
        if st.button("📆 Tomorrow",  width='stretch', key="tomorrow"):
            st.session_state.selected_date = (datetime.now() + timedelta(days=1)).date()
            st.rerun()

    with date_col3:
        if st.button("📆 Day After",  width='stretch', key="dayafter"):
            st.session_state.selected_date = (datetime.now() + timedelta(days=2)).date()
            st.rerun()

    with date_col4:
        custom_date = st.date_input(
            "Pick date",
            value=st.session_state.selected_date,
            label_visibility="collapsed"
        )
        if custom_date != st.session_state.selected_date:
            st.session_state.selected_date = custom_date
            st.rerun()

    selected_date = st.session_state.selected_date
    st.markdown(f"### Games for {selected_date.strftime('%A, %B %d, %Y')}")

    with st.spinner("Fetching games..."):
        games = st.session_state.collector.get_games_for_date(
            datetime.combine(selected_date, datetime.min.time())
        )

    if not games:
        st.info("No games scheduled for this date.")
    else:
        st.write(f"**{len(games)} games found**")

        for i, game in enumerate(games):
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                            padding: 1rem; border-radius: 0.75rem; border: 1px solid #334155; margin-bottom: 0.5rem;">
                    <span style="color: #94a3b8;">✈️</span> 
                    <span style="font-weight: bold; color: #e2e8f0;">{game['away_team']}</span>
                    <span style="color: #64748b; margin: 0 0.5rem;">@</span>
                    <span style="color: #94a3b8;">🏠</span>
                    <span style="font-weight: bold; color: #e2e8f0;">{game['home_team']}</span>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if st.button("🔮 Predict", key=f"predict_{i}",  width='stretch'):
                    if st.session_state.agent and st.session_state.agent.is_ready:
                        with st.spinner("Analyzing..."):
                            result = st.session_state.agent.predict_game(
                                game['home_team'], game['away_team'], is_team1_home=True
                            )
                        if 'error' not in result or not result['error']:
                            st.session_state.current_result = result
                            st.session_state.current_teams = (game['home_team'], game['away_team'])
                    else:
                        st.warning("Initialize the AI agent first!")

        # Show prediction result
        if 'current_result' in st.session_state and st.session_state.current_result:
            st.markdown("---")
            display_prediction(
                st.session_state.current_result,
                st.session_state.current_teams[0],
                st.session_state.current_teams[1]
            )

# Tab 2: Custom Matchup
with tab2:
    st.subheader("Create Custom Matchup")

    teams_df = st.session_state.collector.get_all_teams()
    team_names = sorted(teams_df['full_name'].tolist())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏠 Home Team")
        team1 = st.selectbox("Select home team", team_names, key="custom_team1")

    with col2:
        st.markdown("### ✈️ Away Team")
        team2 = st.selectbox("Select away team", team_names, key="custom_team2", index=1)

    if st.button("🔮 Predict Game Outcome", type="primary",  width='stretch', key="btn_custom_predict"):
        if not st.session_state.agent or not st.session_state.agent.is_ready:
            st.warning("Initialize the AI agent first!")
        elif team1 == team2:
            st.error("Select two different teams")
        else:
            with st.spinner(f"Analyzing {team1} vs {team2}..."):
                result = st.session_state.agent.predict_game(team1, team2, is_team1_home=True)

            if 'error' in result and result['error']:
                st.error(f"Error: {result['error']}")
            else:
                display_prediction(result, team1, team2)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #666; font-size: 0.8rem;">
    Built with Streamlit • ML + Injury Tracking • 
    <a href="https://github.com/sangdelmenon" style="color: #60a5fa;">GitHub</a>
</p>
""", unsafe_allow_html=True)