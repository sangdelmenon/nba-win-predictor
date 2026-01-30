"""
Enhanced Agentic AI for NBA Predictions with Injury Tracking
"""
import numpy as np
from src.model import NBAPredictor, create_training_data
from src.feature_engineering import FeatureEngineer
from src.player_impact import get_injury_tracker, STAR_PLAYERS
from sklearn.model_selection import train_test_split
import pandas as pd


class NBAAgent:
    """
    Autonomous AI Agent for NBA Game Predictions
    - Considers player injuries and star availability
    - Auto-selects best model based on data
    - Provides human-readable explanations
    """

    def __init__(self):
        self.models = {}
        self.best_model_name = None
        self.feature_engineer = FeatureEngineer()
        self.injury_tracker = get_injury_tracker()
        self.is_ready = False
        self.model_scores = {}

    def initialize(self, force_retrain=False):
        """Initialize and train models"""
        print("🤖 Agent initializing...")

        model = NBAPredictor(model_type='xgboost')
        if not force_retrain and model.load('models/nba_predictor.joblib'):
            self.models['xgboost'] = model
            self.best_model_name = 'xgboost'
            self.is_ready = True
            print("✅ Loaded pre-trained model")
            return

        print("📊 Training new models...")
        self._train_all_models()
        self.is_ready = True
        print("✅ Agent ready!")

    def _train_all_models(self):
        """Train multiple models and select the best one"""
        df = create_training_data(5000)
        X = df.drop('team1_wins', axis=1)
        y = df['team1_wins']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_types = ['xgboost', 'random_forest', 'gradient_boosting', 'logistic']

        for model_type in model_types:
            print(f"  Training {model_type}...")
            model = NBAPredictor(model_type=model_type)
            model.train(X_train, y_train)
            results = model.evaluate(X_test, y_test)
            self.models[model_type] = model
            self.model_scores[model_type] = results['accuracy']
            print(f"    Accuracy: {results['accuracy']:.4f}")

        self.best_model_name = max(self.model_scores, key=self.model_scores.get)
        print(f"\n🏆 Best model: {self.best_model_name} ({self.model_scores[self.best_model_name]:.4f})")

        self.models[self.best_model_name].save()

    def set_injury(self, player_name, status, reason=""):
        """Set a player's injury status"""
        self.injury_tracker.set_injury(player_name, status, reason)

    def clear_injury(self, player_name):
        """Clear a player's injury"""
        self.injury_tracker.clear_injury(player_name)

    def refresh_injuries_from_espn(self):
        """Refresh injury data from Basketball Reference"""
        from src.injury_fetcher import update_injury_tracker_from_espn
        return update_injury_tracker_from_espn(self.injury_tracker)

    def get_team_injury_report(self, team_name):
        """Get injury report for a team"""
        return self.injury_tracker.calculate_team_strength(team_name)

    def predict_game(self, team1, team2, is_team1_home=True):
        """Predict game outcome with injury adjustments"""
        if not self.is_ready:
            self.initialize()

        print(f"\n🏀 Analyzing: {team1} vs {team2}")

        try:
            features, team1_stats, team2_stats = self.feature_engineer.create_matchup_features(
                team1, team2, is_team1_home
            )
        except Exception as e:
            return {'error': str(e), 'prediction': None}

        # Get injury adjustments
        team1_injury_adj = self.injury_tracker.get_injury_adjustment(team1)
        team2_injury_adj = self.injury_tracker.get_injury_adjustment(team2)

        # Get injury reports
        _, team1_strength, team1_missing, team1_out = self.injury_tracker.calculate_team_strength(team1)
        _, team2_strength, team2_missing, team2_out = self.injury_tracker.calculate_team_strength(team2)

        # Create feature dataframe
        feature_names = self.feature_engineer.get_feature_names()
        X = pd.DataFrame([features])[feature_names]

        # Get base prediction
        model = self.models[self.best_model_name]
        probabilities = model.predict_proba(X)[0]

        # FIXED: probabilities[0] is team2 (class 0), probabilities[1] is team1 (class 1)
        team1_base_prob = probabilities[1]
        team2_base_prob = probabilities[0]

        print(f"  Base probabilities: {team1} {team1_base_prob:.2%}, {team2} {team2_base_prob:.2%}")
        print(f"  Injury adjustments: {team1} {team1_injury_adj:.2f}x, {team2} {team2_injury_adj:.2f}x")

        # Apply injury adjustments (multiplicative)
        team1_adj_prob = team1_base_prob * team1_injury_adj
        team2_adj_prob = team2_base_prob * team2_injury_adj

        # ENHANCED: Apply win percentage boost for dominant teams
        win_pct_diff = team1_stats.get('win_pct', 0.5) - team2_stats.get('win_pct', 0.5)

        # If there's a big talent gap, boost the better team more
        if abs(win_pct_diff) > 0.15:  # 15% difference is significant
            quality_factor = 0.4  # 40% weight to pure win percentage

            if win_pct_diff > 0:  # Team1 is better
                boost = 0.5 + (win_pct_diff * 1.5)  # Amplify the difference
                team1_adj_prob = team1_adj_prob * (1 - quality_factor) + boost * quality_factor
            else:  # Team2 is better
                boost = 0.5 - (win_pct_diff * 1.5)
                team2_adj_prob = team2_adj_prob * (1 - quality_factor) + boost * quality_factor

        # Renormalize to ensure probabilities sum to 1
        total = team1_adj_prob + team2_adj_prob
        team1_final_prob = team1_adj_prob / total
        team2_final_prob = team2_adj_prob / total

        print(f"  Final probabilities: {team1} {team1_final_prob:.2%}, {team2} {team2_final_prob:.2%}")

        # Determine winner
        prediction = 1 if team1_final_prob > 0.5 else 0
        winner = team1 if prediction == 1 else team2
        win_prob = team1_final_prob if prediction == 1 else team2_final_prob

        # Generate explanation
        explanation = self._generate_explanation(
            team1, team2, team1_stats, team2_stats,
            features, prediction, [team2_final_prob, team1_final_prob],
            is_team1_home, team1_out, team2_out,
            team1_strength, team2_strength
        )

        # Get feature importance
        importance = model.get_feature_importance()
        top_factors = self._get_top_factors(features, importance)

        return {
            'team1': team1,
            'team2': team2,
            'predicted_winner': winner,
            'win_probability': float(win_prob),
            'team1_win_prob': float(team1_final_prob),
            'team2_win_prob': float(team2_final_prob),
            'team1_stats': team1_stats,
            'team2_stats': team2_stats,
            'team1_injuries': team1_out,
            'team2_injuries': team2_out,
            'team1_strength': team1_strength,
            'team2_strength': team2_strength,
            'explanation': explanation,
            'top_factors': top_factors,
            'model_used': self.best_model_name,
            'confidence': self._calculate_confidence([team2_final_prob, team1_final_prob])
        }

    def _generate_explanation(self, team1, team2, team1_stats, team2_stats,
                              features, prediction, probabilities, is_home,
                              team1_injuries, team2_injuries,
                              team1_strength, team2_strength):
        """Generate human-readable explanation with injury info"""
        winner = team1 if prediction == 1 else team2
        confidence = max(probabilities)

        explanations = []

        wp1 = team1_stats.get('win_pct', 0.5)
        wp2 = team2_stats.get('win_pct', 0.5)
        wp_diff = wp1 - wp2

        if abs(wp_diff) > 0.15:
            better_team = team1 if wp_diff > 0 else team2
            explanations.append(
                f"**{better_team}** has a significantly better record "
                f"({max(wp1,wp2)*100:.0f}% vs {min(wp1,wp2)*100:.0f}%)"
            )

        if team1_injuries:
            injury_list = ", ".join([f"{p[0]} ({p[2]})" for p in team1_injuries[:3]])
            explanations.append(f"⚠️ **{team1}** missing key players: {injury_list}")

        if team2_injuries:
            injury_list = ", ".join([f"{p[0]} ({p[2]})" for p in team2_injuries[:3]])
            explanations.append(f"⚠️ **{team2}** missing key players: {injury_list}")

        if abs(team1_strength - team2_strength) > 10:
            stronger = team1 if team1_strength > team2_strength else team2
            explanations.append(
                f"**{stronger}** is at higher effective strength "
                f"({max(team1_strength, team2_strength):.0f}% vs {min(team1_strength, team2_strength):.0f}%)"
            )

        form_diff = features.get('recent_form_diff', 0)
        if abs(form_diff) >= 3:
            hotter_team = team1 if form_diff > 0 else team2
            explanations.append(f"**{hotter_team}** is in much better recent form (last 10 games)")

        if is_home:
            explanations.append(f"**{team1}** has home court advantage (+3-4%)")

        pm_diff = features.get('plus_minus_diff', 0)
        if abs(pm_diff) > 5:
            better_team = team1 if pm_diff > 0 else team2
            explanations.append(f"**{better_team}** has a superior point differential")

        if confidence > 0.75:
            conf_statement = "🔥 **High confidence prediction**"
        elif confidence > 0.6:
            conf_statement = "📊 **Moderate confidence prediction**"
        else:
            conf_statement = "⚖️ **Close matchup - could go either way**"

        explanation = f"### 🏆 Prediction: **{winner}** wins ({confidence*100:.0f}%)\n\n"
        explanation += "**Key Factors:**\n"
        for exp in explanations[:5]:
            explanation += f"- {exp}\n"
        explanation += f"\n{conf_statement}"

        return explanation

    def _get_top_factors(self, features, importance, top_n=5):
        """Get top contributing factors"""
        if importance is None:
            return []

        contributions = []
        for feat, imp in importance.items():
            if feat in features:
                value = features[feat]
                contributions.append({
                    'feature': feat,
                    'importance': imp,
                    'value': value,
                    'contribution': imp * abs(value) if isinstance(value, (int, float)) else imp
                })

        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        return contributions[:top_n]

    def _calculate_confidence(self, probabilities):
        """Calculate confidence level"""
        max_prob = max(probabilities)
        if max_prob > 0.7:
            return 'high'
        elif max_prob > 0.6:
            return 'medium'
        else:
            return 'low'


if __name__ == "__main__":
    agent = NBAAgent()
    agent.initialize(force_retrain=True)

    print("\n" + "=" * 50)
    result = agent.predict_game("Los Angeles Lakers", "Washington Wizards", is_team1_home=True)
    print(result['explanation'])
    print(f"\nLakers: {result['team1_win_prob']:.1%}")
    print(f"Wizards: {result['team2_win_prob']:.1%}")