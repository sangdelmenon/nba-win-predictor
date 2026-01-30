"""
Enhanced Feature Engineering for NBA Game Prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_collector import NBADataCollector


class FeatureEngineer:
    def __init__(self):
        self.collector = NBADataCollector()
    
    def create_matchup_features(self, team1_name, team2_name, is_team1_home=True):
        """
        Create comprehensive features for a matchup between two teams
        """
        team1_id = self.collector.get_team_id(team1_name)
        team2_id = self.collector.get_team_id(team2_name)
        
        if not team1_id or not team2_id:
            raise ValueError(f"Could not find team IDs for {team1_name} or {team2_name}")
        
        # Get team stats (enhanced)
        team1_stats = self.collector.get_team_stats_enhanced(team1_id)
        team2_stats = self.collector.get_team_stats_enhanced(team2_id)
        
        # Get head-to-head
        h2h = self.collector.get_head_to_head(team1_id, team2_id)
        
        # Calculate head-to-head stats
        h2h_stats = self._calculate_h2h_stats(h2h)
        
        # Build comprehensive feature vector
        features = {}
        
        # ============ BASIC STATS ============
        features['team1_win_pct'] = team1_stats.get('win_pct', 0.5)
        features['team1_avg_pts'] = team1_stats.get('avg_pts', 110)
        features['team1_avg_reb'] = team1_stats.get('avg_reb', 44)
        features['team1_avg_ast'] = team1_stats.get('avg_ast', 25)
        features['team1_avg_fg_pct'] = team1_stats.get('avg_fg_pct', 0.46)
        features['team1_avg_fg3_pct'] = team1_stats.get('avg_fg3_pct', 0.36)
        features['team1_avg_ft_pct'] = team1_stats.get('avg_ft_pct', 0.78)
        features['team1_avg_plus_minus'] = team1_stats.get('avg_plus_minus', 0)
        
        features['team2_win_pct'] = team2_stats.get('win_pct', 0.5)
        features['team2_avg_pts'] = team2_stats.get('avg_pts', 110)
        features['team2_avg_reb'] = team2_stats.get('avg_reb', 44)
        features['team2_avg_ast'] = team2_stats.get('avg_ast', 25)
        features['team2_avg_fg_pct'] = team2_stats.get('avg_fg_pct', 0.46)
        features['team2_avg_fg3_pct'] = team2_stats.get('avg_fg3_pct', 0.36)
        features['team2_avg_ft_pct'] = team2_stats.get('avg_ft_pct', 0.78)
        features['team2_avg_plus_minus'] = team2_stats.get('avg_plus_minus', 0)
        
        # ============ RECENT FORM (Last 5 & 10 games) ============
        features['team1_last_5_wins'] = team1_stats.get('last_5_wins', 2.5)
        features['team1_last_10_wins'] = team1_stats.get('last_10_wins', 5)
        features['team1_win_streak'] = team1_stats.get('win_streak', 0)
        features['team1_loss_streak'] = team1_stats.get('loss_streak', 0)
        
        features['team2_last_5_wins'] = team2_stats.get('last_5_wins', 2.5)
        features['team2_last_10_wins'] = team2_stats.get('last_10_wins', 5)
        features['team2_win_streak'] = team2_stats.get('win_streak', 0)
        features['team2_loss_streak'] = team2_stats.get('loss_streak', 0)
        
        # ============ REST & FATIGUE ============
        features['team1_days_rest'] = team1_stats.get('days_rest', 2)
        features['team1_is_back_to_back'] = 1 if team1_stats.get('days_rest', 2) <= 1 else 0
        features['team2_days_rest'] = team2_stats.get('days_rest', 2)
        features['team2_is_back_to_back'] = 1 if team2_stats.get('days_rest', 2) <= 1 else 0
        features['rest_advantage'] = features['team1_days_rest'] - features['team2_days_rest']
        
        # ============ HOME/AWAY SPLITS ============
        features['team1_home_win_pct'] = team1_stats.get('home_win_pct', 0.55)
        features['team1_away_win_pct'] = team1_stats.get('away_win_pct', 0.45)
        features['team2_home_win_pct'] = team2_stats.get('home_win_pct', 0.55)
        features['team2_away_win_pct'] = team2_stats.get('away_win_pct', 0.45)
        
        # ============ OFFENSIVE STATS ============
        features['team1_avg_3pa'] = team1_stats.get('avg_3pa', 35)
        features['team1_offensive_rating'] = team1_stats.get('offensive_rating', 110)
        features['team2_avg_3pa'] = team2_stats.get('avg_3pa', 35)
        features['team2_offensive_rating'] = team2_stats.get('offensive_rating', 110)
        
        # ============ DEFENSIVE STATS ============
        features['team1_avg_stl'] = team1_stats.get('avg_stl', 8)
        features['team1_avg_blk'] = team1_stats.get('avg_blk', 5)
        features['team1_avg_tov'] = team1_stats.get('avg_tov', 14)
        features['team1_defensive_rating'] = team1_stats.get('defensive_rating', 110)
        features['team1_opp_fg_pct'] = team1_stats.get('opp_fg_pct', 0.46)
        
        features['team2_avg_stl'] = team2_stats.get('avg_stl', 8)
        features['team2_avg_blk'] = team2_stats.get('avg_blk', 5)
        features['team2_avg_tov'] = team2_stats.get('avg_tov', 14)
        features['team2_defensive_rating'] = team2_stats.get('defensive_rating', 110)
        features['team2_opp_fg_pct'] = team2_stats.get('opp_fg_pct', 0.46)
        
        # ============ DIFFERENTIAL FEATURES ============
        features['win_pct_diff'] = features['team1_win_pct'] - features['team2_win_pct']
        features['pts_diff'] = features['team1_avg_pts'] - features['team2_avg_pts']
        features['reb_diff'] = features['team1_avg_reb'] - features['team2_avg_reb']
        features['ast_diff'] = features['team1_avg_ast'] - features['team2_avg_ast']
        features['fg_pct_diff'] = features['team1_avg_fg_pct'] - features['team2_avg_fg_pct']
        features['fg3_pct_diff'] = features['team1_avg_fg3_pct'] - features['team2_avg_fg3_pct']
        features['plus_minus_diff'] = features['team1_avg_plus_minus'] - features['team2_avg_plus_minus']
        features['recent_form_diff'] = features['team1_last_10_wins'] - features['team2_last_10_wins']
        features['offensive_rating_diff'] = features['team1_offensive_rating'] - features['team2_offensive_rating']
        features['defensive_rating_diff'] = features['team1_defensive_rating'] - features['team2_defensive_rating']
        
        # Net rating difference (offense - defense for each team)
        team1_net_rating = features['team1_offensive_rating'] - features['team1_defensive_rating']
        team2_net_rating = features['team2_offensive_rating'] - features['team2_defensive_rating']
        features['net_rating_diff'] = team1_net_rating - team2_net_rating
        
        # ============ HEAD-TO-HEAD ============
        features['h2h_win_pct'] = h2h_stats.get('win_pct', 0.5)
        features['h2h_avg_margin'] = h2h_stats.get('avg_margin', 0)
        features['h2h_games_played'] = min(h2h_stats.get('games_played', 0), 10) / 10  # Normalize
        
        # ============ HOME COURT ============
        features['is_home'] = 1 if is_team1_home else 0
        
        # Effective home advantage (team1's home win% vs team2's away win%)
        if is_team1_home:
            features['effective_home_adv'] = features['team1_home_win_pct'] - features['team2_away_win_pct']
        else:
            features['effective_home_adv'] = features['team1_away_win_pct'] - features['team2_home_win_pct']
        
        return features, team1_stats, team2_stats
    
    def _calculate_h2h_stats(self, h2h_df):
        """Calculate head-to-head statistics"""
        if h2h_df.empty:
            return {'win_pct': 0.5, 'avg_margin': 0, 'games_played': 0}
        
        wins = len(h2h_df[h2h_df['WL'] == 'W'])
        games = len(h2h_df)
        
        # Calculate average margin
        if 'PLUS_MINUS' in h2h_df.columns:
            avg_margin = h2h_df['PLUS_MINUS'].mean()
        elif 'PTS' in h2h_df.columns:
            avg_margin = h2h_df['PTS'].mean() - 105  # Rough estimate
        else:
            avg_margin = 0
        
        return {
            'win_pct': wins / games if games > 0 else 0.5,
            'avg_margin': avg_margin,
            'games_played': games
        }
    
    def get_feature_names(self):
        """Return list of feature names in order"""
        return [
            # Basic stats - Team 1
            'team1_win_pct', 'team1_avg_pts', 'team1_avg_reb', 'team1_avg_ast',
            'team1_avg_fg_pct', 'team1_avg_fg3_pct', 'team1_avg_ft_pct', 'team1_avg_plus_minus',
            # Basic stats - Team 2
            'team2_win_pct', 'team2_avg_pts', 'team2_avg_reb', 'team2_avg_ast',
            'team2_avg_fg_pct', 'team2_avg_fg3_pct', 'team2_avg_ft_pct', 'team2_avg_plus_minus',
            # Recent form
            'team1_last_5_wins', 'team1_last_10_wins', 'team1_win_streak', 'team1_loss_streak',
            'team2_last_5_wins', 'team2_last_10_wins', 'team2_win_streak', 'team2_loss_streak',
            # Rest & Fatigue
            'team1_days_rest', 'team1_is_back_to_back', 'team2_days_rest', 'team2_is_back_to_back', 'rest_advantage',
            # Home/Away splits
            'team1_home_win_pct', 'team1_away_win_pct', 'team2_home_win_pct', 'team2_away_win_pct',
            # Offensive stats
            'team1_avg_3pa', 'team1_offensive_rating', 'team2_avg_3pa', 'team2_offensive_rating',
            # Defensive stats
            'team1_avg_stl', 'team1_avg_blk', 'team1_avg_tov', 'team1_defensive_rating', 'team1_opp_fg_pct',
            'team2_avg_stl', 'team2_avg_blk', 'team2_avg_tov', 'team2_defensive_rating', 'team2_opp_fg_pct',
            # Differentials
            'win_pct_diff', 'pts_diff', 'reb_diff', 'ast_diff', 'fg_pct_diff', 'fg3_pct_diff',
            'plus_minus_diff', 'recent_form_diff', 'offensive_rating_diff', 'defensive_rating_diff', 'net_rating_diff',
            # Head-to-head
            'h2h_win_pct', 'h2h_avg_margin', 'h2h_games_played',
            # Home court
            'is_home', 'effective_home_adv'
        ]
