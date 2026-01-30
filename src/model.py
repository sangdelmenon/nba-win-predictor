"""
NBA Game Prediction Model - Uses real data from /data folder
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib


class NBAPredictor:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model with optimized parameters"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=2000,
                C=0.5,
                random_state=42
            )

    def train(self, X, y, tune_hyperparams=False):
        """Train the model"""
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None

        X_scaled = self.scaler.fit_transform(X)

        if tune_hyperparams:
            self._tune_hyperparameters(X_scaled, y)
        else:
            self.model.fit(X_scaled, y)

        self.is_fitted = True

        train_pred = self.model.predict(X_scaled)
        return accuracy_score(y, train_pred)

    def _tune_hyperparameters(self, X, y):
        """Hyperparameter tuning"""
        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.03, 0.05, 0.1]
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            print(f"Best params: {grid_search.best_params_}")
        else:
            self.model.fit(X, y)

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'report': classification_report(y, y_pred, output_dict=True)
        }

    def get_feature_importance(self):
        if not self.is_fitted:
            return None
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            return None

        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return importance

    def save(self, filepath='models/nba_predictor.joblib'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath='models/nba_predictor.joblib'):
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.model_type = data['model_type']
            self.feature_names = data['feature_names']
            self.is_fitted = True
            return True
        return False


def load_team_data():
    """Load team data from /data folder"""
    rosters_path = 'data/rosters.json'

    team_strengths = {}

    if os.path.exists(rosters_path):
        with open(rosters_path, 'r') as f:
            data = json.load(f)

        players = data.get('players', {})

        for player_name, player_data in players.items():
            team = player_data['team']
            impact = player_data['impact']

            if team not in team_strengths:
                team_strengths[team] = {
                    'total_impact': 0,
                    'top_player_impact': 0,
                    'player_count': 0,
                    'avg_ppg': 0,
                    'players': []
                }

            team_strengths[team]['total_impact'] += impact
            team_strengths[team]['player_count'] += 1
            team_strengths[team]['players'].append({
                'name': player_name,
                'impact': impact,
                'ppg': player_data.get('stats', {}).get('ppg', 0)
            })

            if impact > team_strengths[team]['top_player_impact']:
                team_strengths[team]['top_player_impact'] = impact

        for team in team_strengths:
            players = team_strengths[team]['players']
            if players:
                team_strengths[team]['avg_impact'] = team_strengths[team]['total_impact'] / len(players)
                team_strengths[team]['avg_ppg'] = sum(p['ppg'] for p in players) / len(players)
                sorted_players = sorted(players, key=lambda x: x['impact'], reverse=True)
                team_strengths[team]['top5_impact'] = sum(p['impact'] for p in sorted_players[:5])

        print(f"✅ Loaded team strengths for {len(team_strengths)} teams from rosters.json")

    return team_strengths


def create_training_data(n_samples=5000):
    """
    Create training data using REAL team strengths from /data/rosters.json
    """
    np.random.seed(42)

    team_data = load_team_data()

    if not team_data:
        print("⚠️ No roster data found. Run 'python src/roster_updater.py' first.")
        print("   Using synthetic data as fallback...")
        return create_synthetic_training_data(n_samples)

    teams = list(team_data.keys())

    if len(teams) < 10:
        print("⚠️ Not enough teams in roster data. Using synthetic data...")
        return create_synthetic_training_data(n_samples)

    print(f"Creating training data from {len(teams)} teams...")

    data = []

    for _ in range(n_samples):
        team1_name, team2_name = np.random.choice(teams, 2, replace=False)
        team1 = team_data[team1_name]
        team2 = team_data[team2_name]

        max_impact = max(t['top5_impact'] for t in team_data.values())

        team1_strength = team1['top5_impact'] / max_impact
        team2_strength = team2['top5_impact'] / max_impact

        team1_win_pct = np.clip(0.25 + team1_strength * 0.5 + np.random.normal(0, 0.08), 0.15, 0.85)
        team2_win_pct = np.clip(0.25 + team2_strength * 0.5 + np.random.normal(0, 0.08), 0.15, 0.85)

        team1_avg_pts = 100 + team1_strength * 20 + np.random.normal(0, 4)
        team2_avg_pts = 100 + team2_strength * 20 + np.random.normal(0, 4)

        team1_avg_reb = 40 + team1_strength * 8 + np.random.normal(0, 3)
        team2_avg_reb = 40 + team2_strength * 8 + np.random.normal(0, 3)

        team1_avg_ast = 20 + team1_strength * 8 + np.random.normal(0, 2)
        team2_avg_ast = 20 + team2_strength * 8 + np.random.normal(0, 2)

        team1_avg_fg_pct = 0.43 + team1_strength * 0.07 + np.random.normal(0, 0.02)
        team2_avg_fg_pct = 0.43 + team2_strength * 0.07 + np.random.normal(0, 0.02)

        team1_avg_fg3_pct = 0.33 + team1_strength * 0.06 + np.random.normal(0, 0.02)
        team2_avg_fg3_pct = 0.33 + team2_strength * 0.06 + np.random.normal(0, 0.02)

        team1_avg_ft_pct = 0.74 + team1_strength * 0.06 + np.random.normal(0, 0.02)
        team2_avg_ft_pct = 0.74 + team2_strength * 0.06 + np.random.normal(0, 0.02)

        team1_avg_plus_minus = (team1_strength - 0.5) * 12 + np.random.normal(0, 3)
        team2_avg_plus_minus = (team2_strength - 0.5) * 12 + np.random.normal(0, 3)

        team1_last_5_wins = np.random.binomial(5, team1_win_pct)
        team1_last_10_wins = np.random.binomial(10, team1_win_pct)
        team2_last_5_wins = np.random.binomial(5, team2_win_pct)
        team2_last_10_wins = np.random.binomial(10, team2_win_pct)

        team1_win_streak = np.random.geometric(1 - team1_win_pct * 0.6) - 1 if np.random.random() < team1_win_pct else 0
        team1_loss_streak = np.random.geometric(team1_win_pct * 0.6 + 0.3) - 1 if np.random.random() > team1_win_pct else 0
        team2_win_streak = np.random.geometric(1 - team2_win_pct * 0.6) - 1 if np.random.random() < team2_win_pct else 0
        team2_loss_streak = np.random.geometric(team2_win_pct * 0.6 + 0.3) - 1 if np.random.random() > team2_win_pct else 0

        team1_days_rest = np.random.choice([1, 2, 3, 4], p=[0.25, 0.45, 0.2, 0.1])
        team2_days_rest = np.random.choice([1, 2, 3, 4], p=[0.25, 0.45, 0.2, 0.1])
        team1_is_b2b = 1 if team1_days_rest <= 1 else 0
        team2_is_b2b = 1 if team2_days_rest <= 1 else 0
        rest_advantage = team1_days_rest - team2_days_rest

        team1_home_win_pct = team1_win_pct + np.random.uniform(0.04, 0.10)
        team1_away_win_pct = team1_win_pct - np.random.uniform(0.04, 0.10)
        team2_home_win_pct = team2_win_pct + np.random.uniform(0.04, 0.10)
        team2_away_win_pct = team2_win_pct - np.random.uniform(0.04, 0.10)

        team1_off_rating = 105 + team1_strength * 15 + np.random.normal(0, 3)
        team1_def_rating = 115 - team1_strength * 15 + np.random.normal(0, 3)
        team2_off_rating = 105 + team2_strength * 15 + np.random.normal(0, 3)
        team2_def_rating = 115 - team2_strength * 15 + np.random.normal(0, 3)

        team1_avg_stl = 6 + team1_strength * 3 + np.random.normal(0, 1)
        team1_avg_blk = 4 + team1_strength * 2 + np.random.normal(0, 0.8)
        team1_avg_tov = 15 - team1_strength * 4 + np.random.normal(0, 1.5)
        team1_avg_3pa = 30 + team1_strength * 10 + np.random.normal(0, 3)
        team1_opp_fg_pct = 0.48 - team1_strength * 0.04 + np.random.normal(0, 0.015)

        team2_avg_stl = 6 + team2_strength * 3 + np.random.normal(0, 1)
        team2_avg_blk = 4 + team2_strength * 2 + np.random.normal(0, 0.8)
        team2_avg_tov = 15 - team2_strength * 4 + np.random.normal(0, 1.5)
        team2_avg_3pa = 30 + team2_strength * 10 + np.random.normal(0, 3)
        team2_opp_fg_pct = 0.48 - team2_strength * 0.04 + np.random.normal(0, 0.015)

        win_pct_diff = team1_win_pct - team2_win_pct
        pts_diff = team1_avg_pts - team2_avg_pts
        reb_diff = team1_avg_reb - team2_avg_reb
        ast_diff = team1_avg_ast - team2_avg_ast
        fg_pct_diff = team1_avg_fg_pct - team2_avg_fg_pct
        fg3_pct_diff = team1_avg_fg3_pct - team2_avg_fg3_pct
        plus_minus_diff = team1_avg_plus_minus - team2_avg_plus_minus
        recent_form_diff = team1_last_10_wins - team2_last_10_wins
        off_rating_diff = team1_off_rating - team2_off_rating
        def_rating_diff = team1_def_rating - team2_def_rating

        net_rating_diff = (team1_off_rating - team1_def_rating) - (team2_off_rating - team2_def_rating)

        h2h_win_pct = 0.5 + (team1_strength - team2_strength) * 0.25 + np.random.normal(0, 0.1)
        h2h_win_pct = np.clip(h2h_win_pct, 0.2, 0.8)
        h2h_avg_margin = (team1_strength - team2_strength) * 8 + np.random.normal(0, 3)
        h2h_games = np.random.uniform(0.3, 1.0)

        is_home = np.random.binomial(1, 0.5)
        effective_home_adv = (team1_home_win_pct - team2_away_win_pct) if is_home else (team1_away_win_pct - team2_home_win_pct)

        strength_diff = team1_strength - team2_strength

        win_score = (
                0.25 * strength_diff +
                0.15 * win_pct_diff +
                0.12 * (net_rating_diff / 15) +
                0.10 * (recent_form_diff / 10) +
                0.08 * (h2h_win_pct - 0.5) +
                0.08 * (is_home * 0.06) +
                0.06 * (rest_advantage / 4) +
                0.04 * (-team1_is_b2b * 0.04 + team2_is_b2b * 0.04) +
                0.04 * (pts_diff / 15) +
                0.04 * (fg_pct_diff * 3) +
                0.04 * effective_home_adv
        )

        win_prob = 1 / (1 + np.exp(-10 * win_score))
        win_prob = np.clip(win_prob + np.random.normal(0, 0.06), 0.08, 0.92)
        team1_wins = np.random.binomial(1, win_prob)

        data.append({
            'team1_win_pct': team1_win_pct,
            'team1_avg_pts': team1_avg_pts,
            'team1_avg_reb': team1_avg_reb,
            'team1_avg_ast': team1_avg_ast,
            'team1_avg_fg_pct': team1_avg_fg_pct,
            'team1_avg_fg3_pct': team1_avg_fg3_pct,
            'team1_avg_ft_pct': team1_avg_ft_pct,
            'team1_avg_plus_minus': team1_avg_plus_minus,
            'team2_win_pct': team2_win_pct,
            'team2_avg_pts': team2_avg_pts,
            'team2_avg_reb': team2_avg_reb,
            'team2_avg_ast': team2_avg_ast,
            'team2_avg_fg_pct': team2_avg_fg_pct,
            'team2_avg_fg3_pct': team2_avg_fg3_pct,
            'team2_avg_ft_pct': team2_avg_ft_pct,
            'team2_avg_plus_minus': team2_avg_plus_minus,
            'team1_last_5_wins': team1_last_5_wins,
            'team1_last_10_wins': team1_last_10_wins,
            'team1_win_streak': min(team1_win_streak, 10),
            'team1_loss_streak': min(team1_loss_streak, 10),
            'team2_last_5_wins': team2_last_5_wins,
            'team2_last_10_wins': team2_last_10_wins,
            'team2_win_streak': min(team2_win_streak, 10),
            'team2_loss_streak': min(team2_loss_streak, 10),
            'team1_days_rest': team1_days_rest,
            'team1_is_back_to_back': team1_is_b2b,
            'team2_days_rest': team2_days_rest,
            'team2_is_back_to_back': team2_is_b2b,
            'rest_advantage': rest_advantage,
            'team1_home_win_pct': team1_home_win_pct,
            'team1_away_win_pct': team1_away_win_pct,
            'team2_home_win_pct': team2_home_win_pct,
            'team2_away_win_pct': team2_away_win_pct,
            'team1_avg_3pa': team1_avg_3pa,
            'team1_offensive_rating': team1_off_rating,
            'team2_avg_3pa': team2_avg_3pa,
            'team2_offensive_rating': team2_off_rating,
            'team1_avg_stl': team1_avg_stl,
            'team1_avg_blk': team1_avg_blk,
            'team1_avg_tov': team1_avg_tov,
            'team1_defensive_rating': team1_def_rating,
            'team1_opp_fg_pct': team1_opp_fg_pct,
            'team2_avg_stl': team2_avg_stl,
            'team2_avg_blk': team2_avg_blk,
            'team2_avg_tov': team2_avg_tov,
            'team2_defensive_rating': team2_def_rating,
            'team2_opp_fg_pct': team2_opp_fg_pct,
            'win_pct_diff': win_pct_diff,
            'pts_diff': pts_diff,
            'reb_diff': reb_diff,
            'ast_diff': ast_diff,
            'fg_pct_diff': fg_pct_diff,
            'fg3_pct_diff': fg3_pct_diff,
            'plus_minus_diff': plus_minus_diff,
            'recent_form_diff': recent_form_diff,
            'offensive_rating_diff': off_rating_diff,
            'defensive_rating_diff': def_rating_diff,
            'net_rating_diff': net_rating_diff,
            'h2h_win_pct': h2h_win_pct,
            'h2h_avg_margin': h2h_avg_margin,
            'h2h_games_played': h2h_games,
            'is_home': is_home,
            'effective_home_adv': effective_home_adv,
            'team1_wins': team1_wins
        })

    print(f"✅ Created {len(data)} training samples using real team data")
    return pd.DataFrame(data)


def create_synthetic_training_data(n_samples=5000):
    """Fallback: Create purely synthetic training data"""
    print("Creating synthetic training data (fallback)...")

    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        team1_strength = np.random.beta(5, 5)
        team2_strength = np.random.beta(5, 5)

        team1_win_pct = np.clip(team1_strength + np.random.normal(0, 0.1), 0.15, 0.85)
        team2_win_pct = np.clip(team2_strength + np.random.normal(0, 0.1), 0.15, 0.85)

        row = {
            'team1_win_pct': team1_win_pct,
            'team1_avg_pts': 100 + team1_strength * 20 + np.random.normal(0, 5),
            'team1_avg_reb': 40 + team1_strength * 8 + np.random.normal(0, 3),
            'team1_avg_ast': 20 + team1_strength * 8 + np.random.normal(0, 2),
            'team1_avg_fg_pct': 0.43 + team1_strength * 0.07,
            'team1_avg_fg3_pct': 0.33 + team1_strength * 0.06,
            'team1_avg_ft_pct': 0.75 + team1_strength * 0.05,
            'team1_avg_plus_minus': (team1_strength - 0.5) * 10,
            'team2_win_pct': team2_win_pct,
            'team2_avg_pts': 100 + team2_strength * 20 + np.random.normal(0, 5),
            'team2_avg_reb': 40 + team2_strength * 8 + np.random.normal(0, 3),
            'team2_avg_ast': 20 + team2_strength * 8 + np.random.normal(0, 2),
            'team2_avg_fg_pct': 0.43 + team2_strength * 0.07,
            'team2_avg_fg3_pct': 0.33 + team2_strength * 0.06,
            'team2_avg_ft_pct': 0.75 + team2_strength * 0.05,
            'team2_avg_plus_minus': (team2_strength - 0.5) * 10,
            'team1_last_5_wins': np.random.binomial(5, team1_win_pct),
            'team1_last_10_wins': np.random.binomial(10, team1_win_pct),
            'team1_win_streak': 0,
            'team1_loss_streak': 0,
            'team2_last_5_wins': np.random.binomial(5, team2_win_pct),
            'team2_last_10_wins': np.random.binomial(10, team2_win_pct),
            'team2_win_streak': 0,
            'team2_loss_streak': 0,
            'team1_days_rest': 2,
            'team1_is_back_to_back': 0,
            'team2_days_rest': 2,
            'team2_is_back_to_back': 0,
            'rest_advantage': 0,
            'team1_home_win_pct': team1_win_pct + 0.05,
            'team1_away_win_pct': team1_win_pct - 0.05,
            'team2_home_win_pct': team2_win_pct + 0.05,
            'team2_away_win_pct': team2_win_pct - 0.05,
            'team1_avg_3pa': 35,
            'team1_offensive_rating': 105 + team1_strength * 15,
            'team2_avg_3pa': 35,
            'team2_offensive_rating': 105 + team2_strength * 15,
            'team1_avg_stl': 7,
            'team1_avg_blk': 5,
            'team1_avg_tov': 13,
            'team1_defensive_rating': 110 - team1_strength * 10,
            'team1_opp_fg_pct': 0.46,
            'team2_avg_stl': 7,
            'team2_avg_blk': 5,
            'team2_avg_tov': 13,
            'team2_defensive_rating': 110 - team2_strength * 10,
            'team2_opp_fg_pct': 0.46,
            'win_pct_diff': team1_win_pct - team2_win_pct,
            'pts_diff': 0,
            'reb_diff': 0,
            'ast_diff': 0,
            'fg_pct_diff': 0,
            'fg3_pct_diff': 0,
            'plus_minus_diff': 0,
            'recent_form_diff': 0,
            'offensive_rating_diff': 0,
            'defensive_rating_diff': 0,
            'net_rating_diff': (team1_strength - team2_strength) * 10,
            'h2h_win_pct': 0.5,
            'h2h_avg_margin': 0,
            'h2h_games_played': 0.5,
            'is_home': np.random.binomial(1, 0.5),
            'effective_home_adv': 0,
        }

        win_score = 0.3 * (team1_strength - team2_strength) + 0.05 * row['is_home']
        win_prob = 1 / (1 + np.exp(-8 * win_score))
        row['team1_wins'] = np.random.binomial(1, win_prob)

        data.append(row)

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("=" * 60)
    print("NBA PREDICTOR - MODEL TRAINING")
    print("=" * 60)

    df = create_training_data(5000)

    X = df.drop('team1_wins', axis=1)
    y = df['team1_wins']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nFeatures: {len(X.columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("\n" + "-" * 40)
    print("Training XGBoost model...")
    model = NBAPredictor(model_type='xgboost')
    train_acc = model.train(X_train, y_train, tune_hyperparams=True)
    print(f"Training accuracy: {train_acc:.2%}")

    results = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {results['accuracy']:.2%}")

    print("\nTop 10 Feature Importances:")
    importance = model.get_feature_importance()
    if importance:
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat}: {imp:.4f}")

    model.save()
    print("\n✅ Model training complete!")