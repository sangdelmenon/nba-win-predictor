"""
Player Impact System - Tracks star players and injuries
"""
import json
import os
from datetime import datetime


def load_star_players_from_rosters():
    """Load high-impact players from rosters.json"""
    rosters_path = 'data/rosters.json'

    if not os.path.exists(rosters_path):
        print("⚠️ rosters.json not found, using default players")
        return get_default_star_players()

    try:
        with open(rosters_path, 'r') as f:
            data = json.load(f)

        players = data.get('players', {})
        star_players = {}

        # Only include players with impact >= 55
        for player_name, player_data in players.items():
            if player_data['impact'] >= 55:
                star_players[player_name] = {
                    'team': player_data['team'],
                    'impact': player_data['impact'],
                    'position': player_data.get('position', 'G/F')
                }

        print(f"✅ Loaded {len(star_players)} star players from rosters.json")
        return star_players

    except Exception as e:
        print(f"Error loading rosters.json: {e}")
        return get_default_star_players()


def get_default_star_players():
    """Fallback star players if rosters.json not available"""
    return {
        "LeBron James": {"team": "Los Angeles Lakers", "impact": 95, "position": "F"},
        "Anthony Davis": {"team": "Los Angeles Lakers", "impact": 90, "position": "C"},
        "Stephen Curry": {"team": "Golden State Warriors", "impact": 95, "position": "G"},
        "Jayson Tatum": {"team": "Boston Celtics", "impact": 95, "position": "F"},
        "Giannis Antetokounmpo": {"team": "Milwaukee Bucks", "impact": 98, "position": "F"},
        "Nikola Jokic": {"team": "Denver Nuggets", "impact": 98, "position": "C"},
        "Joel Embiid": {"team": "Philadelphia 76ers", "impact": 95, "position": "C"},
    }


# Initialize STAR_PLAYERS
STAR_PLAYERS = load_star_players_from_rosters()


class PlayerImpactTracker:
    """Track player injuries and calculate team impact"""

    def __init__(self):
        self.injuries = {}
        self.load_default_injuries()

    def load_default_injuries(self):
        """Load known current injuries"""
        self.injuries = {}

    def set_injury(self, player_name, status, reason=""):
        """Manually set player injury status"""
        self.injuries[player_name] = {"status": status, "reason": reason}

    def clear_injury(self, player_name):
        """Clear player from injury report"""
        if player_name in self.injuries:
            del self.injuries[player_name]

    def reload_rosters(self):
        """Reload STAR_PLAYERS from rosters.json"""
        global STAR_PLAYERS
        STAR_PLAYERS = load_star_players_from_rosters()
        print(f"✅ Reloaded {len(STAR_PLAYERS)} star players")

    def get_team_players(self, team_name):
        """Get all tracked players for a team"""
        return {
            name: data for name, data in STAR_PLAYERS.items()
            if data['team'].lower() == team_name.lower()
        }

    def calculate_team_strength(self, team_name):
        """
        Calculate team's effective strength based on available players
        Returns: (base_strength, injury_adjusted_strength, missing_impact, missing_players)
        """
        team_players = self.get_team_players(team_name)

        if not team_players:
            return 100, 100, 0, []

        total_impact = sum(p['impact'] for p in team_players.values())
        available_impact = 0
        missing_players = []

        for player_name, player_data in team_players.items():
            injury_info = self.injuries.get(player_name, {})
            status = injury_info.get('status', 'Available')

            if status == 'Out':
                missing_players.append((player_name, player_data['impact'], 'Out'))
            elif status == 'Questionable':
                available_impact += player_data['impact'] * 0.5
                missing_players.append((player_name, player_data['impact'] * 0.5, 'Questionable'))
            elif status == 'Probable':
                available_impact += player_data['impact'] * 0.8
            else:
                available_impact += player_data['impact']

        strength_pct = (available_impact / total_impact * 100) if total_impact > 0 else 100
        missing_impact = total_impact - available_impact

        return 100, strength_pct, missing_impact, missing_players

    def get_injury_adjustment(self, team_name):
        """Get win probability adjustment based on injuries"""
        _, strength_pct, missing_impact, _ = self.calculate_team_strength(team_name)

        adjustment = 1.0 - (missing_impact / 500)
        adjustment = max(0.7, min(1.0, adjustment))

        return adjustment

    def get_matchup_report(self, team1_name, team2_name):
        """Generate injury report for a matchup"""
        report = {
            'team1': {
                'name': team1_name,
                'players': self.get_team_players(team1_name),
                'strength': self.calculate_team_strength(team1_name),
                'adjustment': self.get_injury_adjustment(team1_name)
            },
            'team2': {
                'name': team2_name,
                'players': self.get_team_players(team2_name),
                'strength': self.calculate_team_strength(team2_name),
                'adjustment': self.get_injury_adjustment(team2_name)
            }
        }
        return report


injury_tracker = PlayerImpactTracker()


def get_injury_tracker():
    return injury_tracker


if __name__ == "__main__":
    tracker = PlayerImpactTracker()

    tracker.set_injury("LeBron James", "Out", "Ankle")

    print("=" * 50)
    print("LAKERS STRENGTH")
    print("=" * 50)
    base, adjusted, missing, players = tracker.calculate_team_strength("Los Angeles Lakers")
    print(f"Base: {base}%")
    print(f"Adjusted: {adjusted:.1f}%")
    print(f"Missing impact: {missing}")