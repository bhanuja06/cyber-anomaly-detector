import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_medium_dataset():
    # Define base data
    protocols = ['TCP', 'UDP', 'ICMP']
    response_codes = [200, 301, 302, 400, 401, 403, 404, 500, 502, 503]
    user_agents = ['Chrome/120.0', 'Firefox/115.0', 'Safari/16.0', 'Edge/119.0', 
                   'Mobile Safari', 'Android Browser', 'Bot/2.1', 'Scanner/1.0']
    
    data = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    
    for i in range(500):
        # Add some anomalies (5% of data)
        is_anomaly = random.random() < 0.05
        
        timestamp = start_time + timedelta(minutes=i)
        source_ip = f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}"
        dest_ip = f"10.0.{random.randint(0, 5)}.{random.randint(1, 254)}"
        
        if is_anomaly:
            # Anomalous data
            port = random.choice([22, 23, 3389, 445])  # Suspicious ports
            packet_size = random.randint(60000, 65535)  # Very large packets
            duration = random.randint(300, 600)  # Long duration
            flag_count = random.randint(100, 200)  # High flag count
            login_attempts = random.randint(20, 50)  # Many login attempts
            response_code = random.choice([401, 403, 500, 502])  # Error codes
        else:
            # Normal data
            port = random.choice([80, 443, 53, 123, 161, 443, 8080, 8443])
            packet_size = random.randint(64, 4096)
            duration = random.randint(1, 60)
            flag_count = random.randint(1, 10)
            login_attempts = random.randint(1, 3)
            response_code = random.choice([200, 301, 302])
        
        data.append({
            'timestamp': timestamp,
            'source_ip': source_ip,
            'destination_ip': dest_ip,
            'port': port,
            'protocol': random.choice(protocols),
            'packet_size': packet_size,
            'duration': duration,
            'flag_count': flag_count,
            'login_attempts': login_attempts,
            'response_code': response_code,
            'user_agent': random.choice(user_agents),
            'bytes_sent': random.randint(100, 10000),
            'bytes_received': random.randint(100, 10000),
            'is_encrypted': random.choice([0, 1]),
            'geo_location': random.choice(['US', 'IN', 'UK', 'DE', 'JP', 'CN', 'RU']),
            'threat_score': random.randint(0, 100)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/medium_cyber_data.csv', index=False)
    print(f"Generated medium dataset: {len(df)} records")
    return df

if __name__ == "__main__":
    generate_medium_dataset()