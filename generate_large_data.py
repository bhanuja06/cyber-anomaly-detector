import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_large_dataset():
    print("Generating large dataset (10,000 records)...")
    
    # Define categories
    protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'FTP', 'SSH', 'DNS', 'SMTP']
    attack_types = ['Normal', 'DDoS', 'Port Scan', 'Brute Force', 'Malware', 'SQL Injection', 'XSS']
    countries = ['US', 'IN', 'UK', 'DE', 'FR', 'JP', 'CN', 'RU', 'BR', 'AU', 'CA', 'SG']
    os_list = ['Windows 10', 'Windows 11', 'Linux', 'MacOS', 'Android', 'iOS']
    
    data = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    
    for i in range(10000):
        if i % 1000 == 0:
            print(f"Generated {i} records...")
        
        # Determine if this is an attack (10% attacks)
        is_attack = random.random() < 0.1
        
        timestamp = start_time + timedelta(seconds=i*5)
        source_ip = f"{random.randint(1, 223)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        dest_ip = f"10.0.{random.randint(0, 15)}.{random.randint(1, 254)}"
        
        if is_attack:
            attack_type = random.choice(attack_types[1:])  # Exclude 'Normal'
            
            # Attack patterns
            if attack_type == 'DDoS':
                packet_size = random.randint(1000, 65535)
                duration = 1  # Very short for DDoS
                flag_count = random.randint(50, 200)
                port = random.randint(1, 1024)
            elif attack_type == 'Port Scan':
                packet_size = 64
                duration = random.randint(100, 300)
                flag_count = random.randint(20, 100)
                port = random.randint(1, 65535)
            elif attack_type == 'Brute Force':
                packet_size = random.randint(128, 512)
                duration = random.randint(30, 120)
                flag_count = random.randint(10, 50)
                port = 22  # SSH
            else:
                packet_size = random.randint(512, 4096)
                duration = random.randint(10, 60)
                flag_count = random.randint(5, 20)
                port = random.choice([80, 443, 3306, 5432])
                
            response_code = random.choice([400, 401, 403, 404, 500])
            threat_score = random.randint(70, 100)
            
        else:
            attack_type = 'Normal'
            packet_size = random.randint(64, 4096)
            duration = random.randint(1, 30)
            flag_count = random.randint(1, 10)
            port = random.choice([80, 443, 53, 123, 161, 443, 8080, 8443])
            response_code = random.choice([200, 301, 302])
            threat_score = random.randint(0, 30)
        
        data.append({
            'id': i,
            'timestamp': timestamp,
            'source_ip': source_ip,
            'destination_ip': dest_ip,
            'port': port,
            'protocol': random.choice(protocols),
            'packet_size': packet_size,
            'duration': duration,
            'flag_count': flag_count,
            'login_attempts': random.randint(1, 5) if not is_attack else random.randint(10, 50),
            'response_code': response_code,
            'user_agent': f"Browser/{random.randint(100, 130)}.0",
            'bytes_sent': packet_size * random.randint(1, 10),
            'bytes_received': packet_size * random.randint(1, 5),
            'is_encrypted': 1 if port in [443, 8443, 22] else random.choice([0, 1]),
            'geo_location': random.choice(countries),
            'operating_system': random.choice(os_list),
            'attack_type': attack_type,
            'threat_score': threat_score,
            'anomaly_score': random.uniform(0, 1) if not is_attack else random.uniform(0.7, 1.0),
            'device_type': random.choice(['Server', 'Workstation', 'Mobile', 'IoT', 'Router']),
            'vpn_used': random.choice([0, 1]),
            'tor_used': 1 if is_attack and random.random() < 0.3 else 0
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/large_cyber_data.csv', index=False)
    print(f"Generated large dataset: {len(df)} records, {len(df.columns)} columns")
    print(f"File saved to: data/large_cyber_data.csv")
    print(f"Attack distribution: {df['attack_type'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    generate_large_dataset()