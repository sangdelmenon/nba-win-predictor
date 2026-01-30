from src.data_collector import NBADataCollector

collector = NBADataCollector()

print("=" * 50)
print("LAKERS STATS")
print("=" * 50)
lakers_id = collector.get_team_id("Lakers")
lakers_stats = collector.get_team_stats_enhanced(lakers_id)
for key, value in sorted(lakers_stats.items()):
    print(f"  {key}: {value}")

print("\n" + "=" * 50)
print("WIZARDS STATS")
print("=" * 50)
wizards_id = collector.get_team_id("Wizards")
wizards_stats = collector.get_team_stats_enhanced(wizards_id)
for key, value in sorted(wizards_stats.items()):
    print(f"  {key}: {value}")

print("\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"Lakers Win %: {lakers_stats.get('win_pct', 0):.1%}")
print(f"Wizards Win %: {wizards_stats.get('win_pct', 0):.1%}")
print(f"Lakers Avg Pts: {lakers_stats.get('avg_pts', 0):.1f}")
print(f"Wizards Avg Pts: {wizards_stats.get('avg_pts', 0):.1f}")
print(f"Lakers +/-: {lakers_stats.get('avg_plus_minus', 0):.1f}")
print(f"Wizards +/-: {wizards_stats.get('avg_plus_minus', 0):.1f}")
