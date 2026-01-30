"""
Automatic Injury Data Fetcher - Basketball Reference (Most Reliable)
"""
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime


class InjuryFetcher:
    """Fetch injury data from Basketball Reference"""

    def __init__(self):
        self.injury_url = "https://www.basketball-reference.com/friv/injuries.fcgi"
        self.team_mappings = self._get_team_mappings()

    def _get_team_mappings(self):
        """Map team abbreviations/names"""
        return {
            'LAL': 'Los Angeles Lakers',
            'GSW': 'Golden State Warriors',
            'BOS': 'Boston Celtics',
            'MIL': 'Milwaukee Bucks',
            'DEN': 'Denver Nuggets',
            'PHI': 'Philadelphia 76ers',
            'PHX': 'Phoenix Suns',
            'DAL': 'Dallas Mavericks',
            'LAC': 'Los Angeles Clippers',
            'MIA': 'Miami Heat',
            'NYK': 'New York Knicks',
            'CLE': 'Cleveland Cavaliers',
            'OKC': 'Oklahoma City Thunder',
            'MIN': 'Minnesota Timberwolves',
            'SAC': 'Sacramento Kings',
            'NOP': 'New Orleans Pelicans',
            'MEM': 'Memphis Grizzlies',
            'HOU': 'Houston Rockets',
            'SAS': 'San Antonio Spurs',
            'IND': 'Indiana Pacers',
            'CHI': 'Chicago Bulls',
            'ATL': 'Atlanta Hawks',
            'TOR': 'Toronto Raptors',
            'ORL': 'Orlando Magic',
            'BKN': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets',
            'DET': 'Detroit Pistons',
            'WAS': 'Washington Wizards',
            'POR': 'Portland Trail Blazers',
            'UTA': 'Utah Jazz',
            # Full names
            'Los Angeles Lakers': 'Los Angeles Lakers',
            'Golden State Warriors': 'Golden State Warriors',
            'Boston Celtics': 'Boston Celtics',
            'Milwaukee Bucks': 'Milwaukee Bucks',
            'Denver Nuggets': 'Denver Nuggets',
            'Philadelphia 76ers': 'Philadelphia 76ers',
            'Phoenix Suns': 'Phoenix Suns',
            'Dallas Mavericks': 'Dallas Mavericks',
            'Los Angeles Clippers': 'Los Angeles Clippers',
            'Miami Heat': 'Miami Heat',
            'New York Knicks': 'New York Knicks',
            'Cleveland Cavaliers': 'Cleveland Cavaliers',
            'Oklahoma City Thunder': 'Oklahoma City Thunder',
            'Minnesota Timberwolves': 'Minnesota Timberwolves',
            'Sacramento Kings': 'Sacramento Kings',
            'New Orleans Pelicans': 'New Orleans Pelicans',
            'Memphis Grizzlies': 'Memphis Grizzlies',
            'Houston Rockets': 'Houston Rockets',
            'San Antonio Spurs': 'San Antonio Spurs',
            'Indiana Pacers': 'Indiana Pacers',
            'Chicago Bulls': 'Chicago Bulls',
            'Atlanta Hawks': 'Atlanta Hawks',
            'Toronto Raptors': 'Toronto Raptors',
            'Orlando Magic': 'Orlando Magic',
            'Brooklyn Nets': 'Brooklyn Nets',
            'Charlotte Hornets': 'Charlotte Hornets',
            'Detroit Pistons': 'Detroit Pistons',
            'Washington Wizards': 'Washington Wizards',
            'Portland Trail Blazers': 'Portland Trail Blazers',
            'Utah Jazz': 'Utah Jazz',
        }

    def fetch_all_injuries(self):
        """
        Scrape injury data from Basketball Reference
        Returns: dict of {player_name: {status, reason, team, date}}
        """
        print("🏥 Fetching injury data from Basketball Reference...")

        injuries = {}

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            response = requests.get(self.injury_url, headers=headers, timeout=15)

            if response.status_code != 200:
                print(f"❌ Failed to fetch page (status {response.status_code})")
                return injuries

            soup = BeautifulSoup(response.content, 'html.parser')

            # Basketball Reference uses a simple table structure
            # Find the injuries table
            table = soup.find('table', {'id': 'injuries'})

            if not table:
                # Try finding any table with injury data
                table = soup.find('table')

            if not table:
                print("⚠️  Could not find injury table on page")
                return injuries

            # Find tbody
            tbody = table.find('tbody')
            if not tbody:
                tbody = table

            rows = tbody.find_all('tr')
            print(f"Found {len(rows)} injury entries")

            for row in rows:
                try:
                    cells = row.find_all(['td', 'th'])

                    if len(cells) < 3:
                        continue

                    # Basketball Reference format:
                    # [Team, Player, Date, Status, Description]
                    team_cell = cells[0].get_text(strip=True)
                    player_cell = cells[1].get_text(strip=True)

                    # Get team name
                    team_name = self._normalize_team_name(team_cell)

                    # Get player name (remove any links)
                    player_name = player_cell

                    # Get injury details (varies by column count)
                    if len(cells) >= 4:
                        date = cells[2].get_text(strip=True)
                        status_text = cells[3].get_text(strip=True)
                        description = cells[4].get_text(strip=True) if len(cells) > 4 else ''
                    else:
                        date = ''
                        status_text = cells[2].get_text(strip=True)
                        description = ''

                    # Parse status and reason
                    status = self._parse_status(status_text, description)
                    reason = self._parse_reason(description, status_text)

                    if player_name and team_name:
                        injuries[player_name] = {
                            'status': status,
                            'reason': reason,
                            'team': team_name,
                            'date': date,
                            'detail': description
                        }

                except Exception as e:
                    continue

            print(f"✅ Found {len(injuries)} injured players")
            return injuries

        except Exception as e:
            print(f"❌ Error scraping Basketball Reference: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _normalize_team_name(self, text):
        """Normalize team name from scraped text"""
        text = text.strip()

        # Direct lookup
        if text in self.team_mappings:
            return self.team_mappings[text]

        # Try partial match
        for key, value in self.team_mappings.items():
            if key.lower() in text.lower() or text.lower() in key.lower():
                return value

        return text

    def _parse_status(self, status_text, description):
        """Parse injury status from text"""
        combined = (status_text + ' ' + description).lower()

        if 'out' in combined:
            return 'Out'
        elif 'questionable' in combined:
            return 'Questionable'
        elif 'doubtful' in combined:
            return 'Questionable'
        elif 'probable' in combined:
            return 'Probable'
        elif 'day-to-day' in combined or 'gtd' in combined:
            return 'Questionable'
        else:
            return 'Out'  # Default to Out if unclear

    def _parse_reason(self, description, status_text):
        """Extract injury reason from description"""
        combined = description + ' ' + status_text
        combined_lower = combined.lower()

        # Common injury keywords
        injuries = {
            'knee': 'Knee',
            'ankle': 'Ankle',
            'hamstring': 'Hamstring',
            'back': 'Back',
            'shoulder': 'Shoulder',
            'hip': 'Hip',
            'foot': 'Foot',
            'hand': 'Hand',
            'wrist': 'Wrist',
            'finger': 'Finger',
            'toe': 'Toe',
            'calf': 'Calf',
            'quad': 'Quadriceps',
            'groin': 'Groin',
            'achilles': 'Achilles',
            'concussion': 'Concussion',
            'illness': 'Illness',
            'personal': 'Personal',
            'rest': 'Rest',
            'load management': 'Load Management',
        }

        for keyword, injury_name in injuries.items():
            if keyword in combined_lower:
                return injury_name

        # Return first 30 chars of description if no keyword match
        if description:
            return description[:30]

        return 'Undisclosed'

    def get_injury_summary(self):
        """Get a formatted summary of all injuries"""
        injuries = self.fetch_all_injuries()

        summary = {
            'total': len(injuries),
            'by_status': {'Out': 0, 'Questionable': 0, 'Probable': 0},
            'by_team': {},
            'injuries': injuries
        }

        for player, info in injuries.items():
            status = info['status']
            team = info['team']

            summary['by_status'][status] = summary['by_status'].get(status, 0) + 1

            if team not in summary['by_team']:
                summary['by_team'][team] = []
            summary['by_team'][team].append(player)

        return summary


def update_injury_tracker_from_espn(injury_tracker):
    """
    Update your PlayerImpactTracker with scraped data
    (Function name kept for compatibility)

    Args:
        injury_tracker: PlayerImpactTracker instance

    Returns:
        int: Number of injuries updated
    """
    fetcher = InjuryFetcher()
    injuries = fetcher.fetch_all_injuries()

    if not injuries:
        print("⚠️  No injuries fetched. Keeping existing injury data.")
        return 0

    # Update with scraped data
    updated_count = 0
    for player_name, injury_data in injuries.items():
        injury_tracker.set_injury(
            player_name,
            injury_data['status'],
            injury_data['reason']
        )
        updated_count += 1

    print(f"✅ Updated {updated_count} injuries in tracker")
    return updated_count


if __name__ == "__main__":
    # Test the fetcher
    print("=" * 70)
    print("BASKETBALL REFERENCE INJURY FETCHER - TEST")
    print("=" * 70)

    fetcher = InjuryFetcher()
    summary = fetcher.get_injury_summary()

    print(f"\n📊 Total Injuries: {summary['total']}")
    print(f"   Out: {summary['by_status']['Out']}")
    print(f"   Questionable: {summary['by_status']['Questionable']}")
    print(f"   Probable: {summary['by_status']['Probable']}")

    if summary['total'] == 0:
        print("\n⚠️  No injuries found.")
        print("   Basketball Reference's page structure may have changed.")
        print("   Try manually adding injuries via the Streamlit UI")
    else:
        print("\n" + "=" * 70)
        print("INJURED PLAYERS (sorted by team)")
        print("=" * 70)

        for team, players in sorted(summary['by_team'].items()):
            print(f"\n{team}:")
            for player in players:
                info = summary['injuries'][player]
                date_str = f" ({info['date']})" if info.get('date') else ""
                print(f"  • {player:<25} - {info['status']:<12} - {info['reason']}{date_str}")