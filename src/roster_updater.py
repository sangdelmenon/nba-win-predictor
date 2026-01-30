"""
NBA Roster Updater - Auto-calculates player impact from real stats
No manual tier assignment needed!
"""
import json
import os
import time
from datetime import datetime
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    commonteamroster,
    playercareerstats,
    leaguedashplayerstats
)


def fetch_all_player_stats():
    """Fetch current season stats for all players"""
    print("Fetching player stats for current season...")
    
    try:
        time.sleep(1)
        
        # Get all player stats for current season
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2024-25',
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame'
        )
        
        df = stats.get_data_frames()[0]
        print(f"✅ Fetched stats for {len(df)} players")
        
        return df
    
    except Exception as e:
        print(f"Error fetching player stats: {e}")
        return None


def calculate_player_impact(row):
    """
    Calculate player impact (0-100) based on real stats
    
    Formula considers:
    - Points per game (weighted heavily)
    - Assists per game
    - Rebounds per game
    - Steals + Blocks (defense)
    - Minutes per game (usage/importance)
    - Efficiency (FG%, TS%)
    - Plus/Minus
    """
    
    # Get stats with defaults
    ppg = row.get('PTS', 0) or 0
    apg = row.get('AST', 0) or 0
    rpg = row.get('REB', 0) or 0
    spg = row.get('STL', 0) or 0
    bpg = row.get('BLK', 0) or 0
    mpg = row.get('MIN', 0) or 0
    fg_pct = row.get('FG_PCT', 0) or 0
    plus_minus = row.get('PLUS_MINUS', 0) or 0
    games = row.get('GP', 1) or 1
    
    # Normalize each stat to 0-100 scale based on NBA ranges
    # Points: 0-35 PPG -> 0-100
    pts_score = min(100, (ppg / 35) * 100)
    
    # Assists: 0-12 APG -> 0-100
    ast_score = min(100, (apg / 12) * 100)
    
    # Rebounds: 0-15 RPG -> 0-100
    reb_score = min(100, (rpg / 15) * 100)
    
    # Defense (steals + blocks): 0-5 -> 0-100
    def_score = min(100, ((spg + bpg) / 5) * 100)
    
    # Minutes: 0-38 MPG -> 0-100 (indicates team reliance)
    min_score = min(100, (mpg / 38) * 100)
    
    # Efficiency: FG% 0.35-0.55 -> 0-100
    eff_score = min(100, max(0, ((fg_pct - 0.35) / 0.20) * 100))
    
    # Plus/Minus: -10 to +10 -> 0-100
    pm_score = min(100, max(0, ((plus_minus + 10) / 20) * 100))
    
    # Games played factor (penalize players who haven't played much)
    games_factor = min(1.0, games / 20)
    
    # Weighted combination
    impact = (
        pts_score * 0.30 +      # Points most important
        ast_score * 0.15 +      # Playmaking
        reb_score * 0.12 +      # Rebounding
        def_score * 0.10 +      # Defense
        min_score * 0.15 +      # Usage/minutes
        eff_score * 0.08 +      # Efficiency
        pm_score * 0.10         # Overall impact
    ) * games_factor
    
    # Scale to 30-98 range (no one gets 100, min 30 for NBA player)
    impact = 30 + (impact * 0.68)
    
    return round(impact)


def fetch_team_rosters():
    """Fetch all team rosters"""
    all_teams = teams.get_teams()
    rosters = {}
    
    print("\nFetching team rosters...")
    
    for team in all_teams:
        try:
            time.sleep(0.5)
            roster = commonteamroster.CommonTeamRoster(team_id=team['id'])
            df = roster.get_data_frames()[0]
            
            rosters[team['full_name']] = {
                'team_id': team['id'],
                'players': df['PLAYER'].tolist(),
                'player_ids': df['PLAYER_ID'].tolist()
            }
            
            print(f"  {team['full_name']}: {len(df)} players")
            
        except Exception as e:
            print(f"  Error fetching {team['full_name']}: {e}")
    
    return rosters


def build_player_database():
    """Build complete player database with auto-calculated impacts"""
    
    print("=" * 60)
    print("NBA ROSTER & IMPACT UPDATER")
    print("=" * 60)
    
    # Step 1: Fetch all player stats
    stats_df = fetch_all_player_stats()
    
    if stats_df is None:
        print("❌ Failed to fetch player stats")
        return None
    
    # Step 2: Fetch team rosters
    rosters = fetch_team_rosters()
    
    # Step 3: Build player database
    print("\nCalculating player impacts...")
    
    players = {}
    
    for _, row in stats_df.iterrows():
        player_name = row['PLAYER_NAME']
        team_name = row['TEAM_NAME'] if 'TEAM_NAME' in row else None
        
        # Calculate impact from stats
        impact = calculate_player_impact(row)
        
        # Get position
        # The API doesn't always give position, so we'll estimate from stats
        ppg = row.get('PTS', 0) or 0
        apg = row.get('AST', 0) or 0
        rpg = row.get('REB', 0) or 0
        
        if rpg > 8:
            position = 'C/F'
        elif apg > 6:
            position = 'G'
        elif ppg > 15 and rpg > 5:
            position = 'F'
        else:
            position = 'G/F'
        
        # Find team from roster if not in stats
        if not team_name:
            for team, roster_data in rosters.items():
                if player_name in roster_data['players']:
                    team_name = team
                    break
        
        if team_name:
            players[player_name] = {
                'team': team_name,
                'impact': impact,
                'position': position,
                'stats': {
                    'ppg': round(row.get('PTS', 0) or 0, 1),
                    'apg': round(row.get('AST', 0) or 0, 1),
                    'rpg': round(row.get('REB', 0) or 0, 1),
                    'mpg': round(row.get('MIN', 0) or 0, 1),
                    'fg_pct': round((row.get('FG_PCT', 0) or 0) * 100, 1),
                    'games': row.get('GP', 0) or 0,
                }
            }
    
    # Sort by impact and show top players
    print("\n" + "=" * 60)
    print("TOP 30 PLAYERS BY CALCULATED IMPACT")
    print("=" * 60)
    
    sorted_players = sorted(players.items(), key=lambda x: x[1]['impact'], reverse=True)
    
    for i, (name, data) in enumerate(sorted_players[:30], 1):
        stats = data['stats']
        print(f"{i:2}. {name:<25} | {data['team']:<25} | Impact: {data['impact']:2} | "
              f"{stats['ppg']}ppg {stats['apg']}apg {stats['rpg']}rpg")
    
    return players


def save_player_database(players, filepath='data/rosters.json'):
    """Save player database to JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = {
        'last_updated': datetime.now().isoformat(),
        'total_players': len(players),
        'players': players
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Saved {len(players)} players to {filepath}")
    
    # Also save a summary by team
    teams_summary = {}
    for name, info in players.items():
        team = info['team']
        if team not in teams_summary:
            teams_summary[team] = []
        teams_summary[team].append({
            'name': name,
            'impact': info['impact'],
            'ppg': info['stats']['ppg']
        })
    
    # Sort each team's players by impact
    for team in teams_summary:
        teams_summary[team] = sorted(teams_summary[team], key=lambda x: x['impact'], reverse=True)
    
    with open('data/teams_summary.json', 'w') as f:
        json.dump(teams_summary, f, indent=2)
    
    print(f"✅ Saved team summaries to data/teams_summary.json")


def update_rosters():
    """Main function to update all rosters"""
    players = build_player_database()
    
    if players:
        save_player_database(players)
        
        # Show some team examples
        print("\n" + "=" * 60)
        print("SAMPLE TEAM ROSTERS")
        print("=" * 60)
        
        sample_teams = ["Los Angeles Lakers", "Golden State Warriors", "Boston Celtics"]
        
        for team in sample_teams:
            print(f"\n{team}:")
            team_players = [(n, d) for n, d in players.items() if d['team'] == team]
            team_players = sorted(team_players, key=lambda x: x[1]['impact'], reverse=True)[:5]
            
            for name, data in team_players:
                print(f"  {name}: Impact {data['impact']} ({data['stats']['ppg']}ppg)")
        
        return players
    
    return None


if __name__ == "__main__":
    update_rosters()
