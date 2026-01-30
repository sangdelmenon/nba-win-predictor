"""
NBA Data Collector - Enhanced with more statistics
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    scoreboardv2,
    teamestimatedmetrics
)


class NBADataCollector:
    def __init__(self):
        self.teams_df = pd.DataFrame(teams.get_teams())
        self.current_season = "2024-25"
        
    def get_all_teams(self):
        """Get all NBA teams"""
        return self.teams_df[['id', 'full_name', 'abbreviation', 'nickname', 'city']]
    
    def get_team_id(self, team_name):
        """Get team ID from name or abbreviation"""
        team = self.teams_df[
            (self.teams_df['full_name'].str.lower() == team_name.lower()) |
            (self.teams_df['abbreviation'].str.lower() == team_name.lower()) |
            (self.teams_df['nickname'].str.lower() == team_name.lower())
        ]
        if len(team) > 0:
            return team.iloc[0]['id']
        return None
    
    def get_team_name(self, team_id):
        """Get team name from ID"""
        team = self.teams_df[self.teams_df['id'] == team_id]
        if len(team) > 0:
            return team.iloc[0]['full_name']
        return None
    
    def get_team_abbrev(self, team_id):
        """Get team abbreviation from ID"""
        team = self.teams_df[self.teams_df['id'] == team_id]
        if len(team) > 0:
            return team.iloc[0]['abbreviation']
        return None
    
    def get_games_for_date(self, game_date):
        """Get all games scheduled for a specific date"""
        try:
            time.sleep(0.6)
            date_str = game_date.strftime('%m/%d/%Y')
            
            scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
            games_df = scoreboard.get_data_frames()[0]
            
            if games_df.empty:
                return []
            
            games = []
            for _, game in games_df.iterrows():
                home_team_id = game.get('HOME_TEAM_ID')
                away_team_id = game.get('VISITOR_TEAM_ID')
                
                games.append({
                    'game_id': game.get('GAME_ID'),
                    'game_date': game_date,
                    'game_time': game.get('GAME_STATUS_TEXT', 'TBD'),
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_team': self.get_team_name(home_team_id),
                    'away_team': self.get_team_name(away_team_id),
                    'home_abbrev': self.get_team_abbrev(home_team_id),
                    'away_abbrev': self.get_team_abbrev(away_team_id),
                })
            
            return games
        except Exception as e:
            print(f"Error fetching games for {game_date}: {e}")
            return []
    
    def get_today_games(self):
        """Get today's games"""
        return self.get_games_for_date(datetime.now())
    
    def get_tomorrow_games(self):
        """Get tomorrow's games"""
        return self.get_games_for_date(datetime.now() + timedelta(days=1))
    
    def get_team_game_log(self, team_id, season=None, last_n_games=30):
        """Get recent games for a team"""
        if season is None:
            season = self.current_season
        
        try:
            time.sleep(0.6)
            game_log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            df = game_log.get_data_frames()[0]
            return df.head(last_n_games)
        except Exception as e:
            print(f"Error fetching team game log: {e}")
            return pd.DataFrame()
    
    def get_head_to_head(self, team1_id, team2_id, last_n_seasons=3):
        """Get head-to-head matchups between two teams"""
        try:
            time.sleep(0.6)
            games = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team1_id,
                vs_team_id_nullable=team2_id,
                season_type_nullable='Regular Season'
            )
            df = games.get_data_frames()[0]
            return df.head(20)
        except Exception as e:
            print(f"Error fetching head-to-head: {e}")
            return pd.DataFrame()
    
    def get_team_stats(self, team_id, season=None):
        """Get basic team statistics (backward compatibility)"""
        return self.get_team_stats_enhanced(team_id, season)
    
    def get_team_stats_enhanced(self, team_id, season=None):
        """Get comprehensive team statistics"""
        game_log = self.get_team_game_log(team_id, season, last_n_games=30)
        
        if game_log.empty:
            return self._get_default_stats()
        
        cols = game_log.columns.tolist()
        stats = {}
        
        # ============ BASIC STATS ============
        stats['games_played'] = len(game_log)
        stats['wins'] = len(game_log[game_log['WL'] == 'W'])
        stats['losses'] = len(game_log[game_log['WL'] == 'L'])
        stats['win_pct'] = stats['wins'] / stats['games_played'] if stats['games_played'] > 0 else 0.5
        
        # Scoring
        stats['avg_pts'] = game_log['PTS'].mean() if 'PTS' in cols else 110
        stats['avg_reb'] = game_log['REB'].mean() if 'REB' in cols else 44
        stats['avg_ast'] = game_log['AST'].mean() if 'AST' in cols else 25
        stats['avg_stl'] = game_log['STL'].mean() if 'STL' in cols else 8
        stats['avg_blk'] = game_log['BLK'].mean() if 'BLK' in cols else 5
        stats['avg_tov'] = game_log['TOV'].mean() if 'TOV' in cols else 14
        
        # Shooting
        stats['avg_fg_pct'] = game_log['FG_PCT'].mean() if 'FG_PCT' in cols else 0.46
        stats['avg_fg3_pct'] = game_log['FG3_PCT'].mean() if 'FG3_PCT' in cols else 0.36
        stats['avg_ft_pct'] = game_log['FT_PCT'].mean() if 'FT_PCT' in cols else 0.78
        stats['avg_3pa'] = game_log['FG3A'].mean() if 'FG3A' in cols else 35
        
        # Plus/Minus
        if 'PLUS_MINUS' in cols:
            stats['avg_plus_minus'] = game_log['PLUS_MINUS'].mean()
        else:
            stats['avg_plus_minus'] = 0
        
        # ============ RECENT FORM ============
        recent_5 = game_log.head(5)
        recent_10 = game_log.head(10)
        stats['last_5_wins'] = len(recent_5[recent_5['WL'] == 'W'])
        stats['last_10_wins'] = len(recent_10[recent_10['WL'] == 'W'])
        
        # Win/Loss streaks
        stats['win_streak'] = self._calculate_streak(game_log, 'W')
        stats['loss_streak'] = self._calculate_streak(game_log, 'L')
        
        # ============ REST & FATIGUE ============
        if 'GAME_DATE' in cols:
            try:
                game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])
                last_game_date = game_log['GAME_DATE'].iloc[0]
                stats['days_rest'] = (datetime.now() - last_game_date).days
                
                # Check for back-to-back
                if len(game_log) >= 2:
                    prev_game_date = game_log['GAME_DATE'].iloc[1]
                    days_between = (last_game_date - prev_game_date).days
                    stats['is_back_to_back'] = 1 if days_between <= 1 else 0
                else:
                    stats['is_back_to_back'] = 0
            except:
                stats['days_rest'] = 2
                stats['is_back_to_back'] = 0
        else:
            stats['days_rest'] = 2
            stats['is_back_to_back'] = 0
        
        # ============ HOME/AWAY SPLITS ============
        if 'MATCHUP' in cols:
            home_games = game_log[game_log['MATCHUP'].str.contains('vs.', na=False)]
            away_games = game_log[game_log['MATCHUP'].str.contains('@', na=False)]
            
            home_wins = len(home_games[home_games['WL'] == 'W'])
            away_wins = len(away_games[away_games['WL'] == 'W'])
            
            stats['home_wins'] = home_wins
            stats['away_wins'] = away_wins
            stats['home_win_pct'] = home_wins / len(home_games) if len(home_games) > 0 else 0.55
            stats['away_win_pct'] = away_wins / len(away_games) if len(away_games) > 0 else 0.45
            
            # Home/Away scoring
            stats['home_avg_pts'] = home_games['PTS'].mean() if len(home_games) > 0 and 'PTS' in cols else stats['avg_pts']
            stats['away_avg_pts'] = away_games['PTS'].mean() if len(away_games) > 0 and 'PTS' in cols else stats['avg_pts']
        else:
            stats['home_wins'] = stats['wins'] // 2
            stats['away_wins'] = stats['wins'] // 2
            stats['home_win_pct'] = 0.55
            stats['away_win_pct'] = 0.45
            stats['home_avg_pts'] = stats['avg_pts'] + 3
            stats['away_avg_pts'] = stats['avg_pts'] - 3
        
        # ============ OFFENSIVE/DEFENSIVE RATINGS ============
        # Estimate offensive rating (points per 100 possessions)
        # Possessions ≈ FGA - OREB + TOV + 0.4 * FTA
        if all(col in cols for col in ['FGA', 'OREB', 'TOV', 'FTA', 'PTS']):
            possessions = game_log['FGA'] - game_log['OREB'] + game_log['TOV'] + 0.4 * game_log['FTA']
            possessions = possessions.replace(0, 1)  # Avoid division by zero
            off_ratings = (game_log['PTS'] / possessions) * 100
            stats['offensive_rating'] = off_ratings.mean()
        else:
            stats['offensive_rating'] = 110 + (stats['avg_pts'] - 110) * 0.5
        
        # Estimate defensive rating (opponent points allowed)
        # We don't have opponent stats directly, so estimate from plus/minus
        stats['defensive_rating'] = stats['offensive_rating'] - stats['avg_plus_minus']
        
        # Opponent FG% (estimate)
        stats['opp_fg_pct'] = 0.46 - (stats['avg_plus_minus'] / 200)  # Rough estimate
        stats['opp_fg_pct'] = max(0.40, min(0.52, stats['opp_fg_pct']))  # Clamp to realistic range
        
        return stats
    
    def _calculate_streak(self, game_log, streak_type='W'):
        """Calculate current win or loss streak"""
        if game_log.empty or 'WL' not in game_log.columns:
            return 0
        
        streak = 0
        for _, game in game_log.iterrows():
            if game['WL'] == streak_type:
                streak += 1
            else:
                break
        return streak
    
    def _get_default_stats(self):
        """Return default stats when data is unavailable"""
        return {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'win_pct': 0.5,
            'avg_pts': 110,
            'avg_reb': 44,
            'avg_ast': 25,
            'avg_stl': 8,
            'avg_blk': 5,
            'avg_tov': 14,
            'avg_fg_pct': 0.46,
            'avg_fg3_pct': 0.36,
            'avg_ft_pct': 0.78,
            'avg_3pa': 35,
            'avg_plus_minus': 0,
            'last_5_wins': 2.5,
            'last_10_wins': 5,
            'win_streak': 0,
            'loss_streak': 0,
            'days_rest': 2,
            'is_back_to_back': 0,
            'home_wins': 0,
            'away_wins': 0,
            'home_win_pct': 0.55,
            'away_win_pct': 0.45,
            'home_avg_pts': 113,
            'away_avg_pts': 107,
            'offensive_rating': 110,
            'defensive_rating': 110,
            'opp_fg_pct': 0.46,
        }
