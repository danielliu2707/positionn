import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats, commonplayerinfo
from nba_api.stats.library.http import NBAStatsHTTP
import time
from datetime import datetime
import os
import requests
from requests.exceptions import Timeout, ConnectionError
import random
from multiprocessing import Pool, cpu_count

# Configure NBA API settings
NBAStatsHTTP.TIMEOUT = 60  # Increase global timeout to 60 seconds
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

def retry_with_exponential_backoff(func, max_retries=5, initial_delay=2):
    """Retry a function with exponential backoff"""
    for retry in range(max_retries):
        try:
            return func()
        except (Timeout, ConnectionError) as e:
            if retry == max_retries - 1:  # Last retry
                raise e
            
            # Calculate delay with some randomness to avoid API congestion
            delay = (initial_delay * (2 ** retry)) + random.uniform(0, 2)
            print(f"Request failed, retrying in {delay:.1f} seconds...")
            time.sleep(delay)
    return None

def download_player_headshot(player_id, player_name):
    """Download player headshot from NBA.com"""
    url = f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"
    output_dir = "player_headshots"
    output_path = f"{output_dir}/{player_id}.png"
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Headshot already exists for {player_name}")
        return
    
    try:
        response = session.get(url, timeout=20)  # Increased timeout, using session
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded headshot for {player_name}")
        else:
            print(f"Could not find headshot for {player_name}")
    except Exception as e:
        print(f"Error downloading headshot for {player_name}: {str(e)}")

def get_active_players():
    """Get all active NBA players"""
    return [player for player in players.get_players() if player['is_active']]

def get_player_info(player_id):
    """Get detailed player information including position"""
    def fetch_info():
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=60)
        time.sleep(2.5)  # Increased delay between requests
        player_info = info.get_data_frames()[0]
        if not player_info.empty:
            basic_pos = player_info.iloc[0]['POSITION']
            pos_map = {
                'Forward-Guard': 'F',
                'Guard-Forward': 'G',
                'Forward-Center': 'F',
                'Center-Forward': 'C',
                'Forward': 'F',
                'Center': 'C',
                'Guard': 'G',
                'F-G': 'F',
                'G-F': 'G',
                'F-C': 'F',
                'C-F': 'C',
                'F': 'F',
                'C': 'C',
                'G': 'G'
            }
            return pos_map.get(basic_pos, basic_pos)
        return ''
    
    return retry_with_exponential_backoff(fetch_info)

def get_player_stats(player_id):
    """Get career statistics for a player"""
    def fetch_stats():
        career = playercareerstats.PlayerCareerStats(player_id=player_id, timeout=60)
        time.sleep(2.5)  # Increased delay between requests
        return career.get_data_frames()[0]
    
    return retry_with_exponential_backoff(fetch_stats)

def convert_season_to_year(season_id):
    """Convert season ID (e.g., '2020-21') to end year (2021)"""
    start_year = int(season_id.split('-')[0])
    return start_year + 1

def process_player_stats(stats_df, player_name, player_position):
    """Process player statistics to match your required format"""
    # Create a copy to avoid SettingWithCopyWarning
    df = stats_df.copy()
    
    # Add player name and position
    df['player'] = player_name
    df['pos'] = player_position
    
    # Convert numeric columns to float
    numeric_cols = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'GP']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate per game statistics
    df['PTS'] = (df['PTS'] / df['GP']).round(2)
    df['AST'] = (df['AST'] / df['GP']).round(2)
    df['REB'] = (df['REB'] / df['GP']).round(2)
    df['STL'] = (df['STL'] / df['GP']).round(2)
    df['BLK'] = (df['BLK'] / df['GP']).round(2)
    df['TOV'] = (df['TOV'] / df['GP']).round(2)
    
    # Calculate advanced metrics
    df['AST_TO'] = df['AST'] / df['TOV']  # AST/TO ratio
    
    df['STOCKS'] = (df['STL'] + df['BLK']).round(2)  # Steals + Blocks
    df['FIC'] = (df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - df['TOV']).round(2)  # Floor Impact Counter
    
    # Convert season ID to full year
    df['year'] = df['SEASON_ID'].apply(convert_season_to_year)
    df['age'] = pd.to_numeric(df['PLAYER_AGE'], errors='coerce')
    
    # Filter for seasons between 2015-2024
    df = df[df['year'].between(2015, 2024)]
    
    # Select and reorder final columns
    result = df[[
        'player',
        'pos',
        'PTS',
        'AST',
        'REB',
        'STL',
        'BLK',
        'TOV',
        'AST_TO',
        'STOCKS',
        'FIC',
        'age',
        'year'
    ]].copy()
    
    # Replace any NaN values with 0
    result = result.fillna(0)
    
    return result

def process_single_player(player):
    """Process a single player's stats"""
    print(f"Processing player: {player['full_name']}")
    try:
        # Download player headshot
        download_player_headshot(player['id'], player['full_name'])
        
        # Get player position from detailed info
        position = get_player_info(player['id'])
        if position is None:
            print(f"Skipping {player['full_name']} due to API errors")
            return None
        
        # Get career stats
        stats = get_player_stats(player['id'])
        if stats is None:
            print(f"Skipping {player['full_name']} due to API errors")
            return None
        
        processed_stats = process_player_stats(
            stats,
            player['full_name'],
            position
        )
        
        print(f"Successfully processed {player['full_name']}")
        return processed_stats
        
    except Exception as e:
        print(f"Error processing player {player['full_name']}: {str(e)}")
        return None

def process_player_batch(players_batch, batch_num):
    """Process a batch of players in parallel and save to a separate file"""
    # Determine number of processes (use 75% of available CPU cores)
    num_processes = max(1, int(cpu_count() * 0.75))
    print(f"Using {num_processes} processes for parallel processing")
    
    # Create a pool of workers
    with Pool(processes=num_processes) as pool:
        # Process players in parallel
        results = pool.map(process_single_player, players_batch)
        
        # Filter out None results (failed processing)
        all_stats = [r for r in results if r is not None]
    
    # Save batch to CSV if we have data
    if all_stats:
        final_df = pd.concat(all_stats, ignore_index=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        output_file = f'data/nba_current_player_stats_{timestamp}_batch{batch_num}.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\nBatch {batch_num} saved to {output_file}")
        print(f"Batch {batch_num} Summary:")
        print(f"Players processed: {len(all_stats)} out of {len(players_batch)}")
        print(f"Total rows: {len(final_df)}")
        print(f"Year range: {final_df['year'].min()} - {final_df['year'].max()}\n")

def main():
    # Get all active players
    print("Fetching active players...")
    active_players = get_active_players()
    total_players = len(active_players)
    print(f"Found {total_players} active players")
    
    # Process all players in batches
    batch_size = 200
    total_batches = (total_players + batch_size - 1) // batch_size
    
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_players)
        batch = active_players[start_idx:end_idx]
        print(f"\nProcessing batch {batch_num} of {total_batches} ({len(batch)} players)")
        process_player_batch(batch, batch_num)

if __name__ == "__main__":
    main() 