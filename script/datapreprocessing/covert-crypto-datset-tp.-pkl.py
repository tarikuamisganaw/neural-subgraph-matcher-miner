import pandas as pd  
import networkx as nx  
import pickle  
  
def convert_token_transfers_to_pkl(csv_path, output_pkl, sample_size=50000):  
    """  
    Convert token_transfers.csv to graph format with PROPER edge labels for visualization.  
    """  
    print(f"Processing {csv_path}...")  
      
    # Create directed graph for token transfers  
    G = nx.DiGraph()  
    processed = 0  
      
    # Stream and process the CSV directly
    chunk_size = 10000  
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, nrows=sample_size):  
        for _, row in chunk.iterrows():  
            from_addr = row.get('from_address')  
            to_addr = row.get('to_address')  
              
            if pd.notna(from_addr) and pd.notna(to_addr) and from_addr != to_addr:  
                # Convert to string to ensure consistency
                from_addr = str(from_addr).lower()
                to_addr = str(to_addr).lower()
                
                # Add nodes with wallet attributes  
                if from_addr not in G:  
                    G.add_node(from_addr, node_type='wallet', label='address')  
                if to_addr not in G:  
                    G.add_node(to_addr, node_type='wallet', label='address')  
                  
                # ✅ FIX: Add MEANINGFUL edge labels that the visualizer expects
                edge_attrs = {
                    'weight': 1.0,
                    'label': 'token_transfer',  # This is what the visualizer looks for!
                    'type': 'transaction',      # Alternative label the visualizer accepts
                    'edge_type': 'token_transfer'
                }  
                
                # Add available transaction data
                if 'value' in row and pd.notna(row['value']):  
                    amount = float(row['value'])
                    edge_attrs['amount'] = amount
                    edge_attrs['weight'] = amount
                    
                    # ✅ ENHANCEMENT: Create more descriptive labels based on amount
                    if amount > 10000:
                        edge_attrs['label'] = 'large_transfer'
                    elif amount > 1000:
                        edge_attrs['label'] = 'medium_transfer'
                    else:
                        edge_attrs['label'] = 'small_transfer'
                    
                if 'time_stamp' in row and pd.notna(row['time_stamp']):  
                    edge_attrs['timestamp'] = int(row['time_stamp'])
                    
                if 'contract_address' in row and pd.notna(row['contract_address']):  
                    token_contract = str(row['contract_address'])
                    edge_attrs['token'] = token_contract
                    
                    # ✅ ENHANCEMENT: Add token-specific labels
                    # Common token contract addresses (you can expand this)
                    token_mapping = {
                        '0xdac17f958d2ee523a2206206994597c13d831ec7': 'USDT',
                        '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': 'USDC', 
                        '0x6b175474e89094c44da98b954eedeac495271d0f': 'DAI'
                    }
                    
                    if token_contract.lower() in token_mapping:
                        token_name = token_mapping[token_contract.lower()]
                        edge_attrs['label'] = f'{token_name}_transfer'
                        edge_attrs['type'] = token_name
                    
                if 'block_number' in row and pd.notna(row['block_number']):  
                    edge_attrs['block'] = int(row['block_number'])
                  
                G.add_edge(from_addr, to_addr, **edge_attrs)  
              
            processed += 1  
            if processed % 10000 == 0:  
                print(f"Processed {processed} transactions...")  
          
        if processed >= sample_size:  
            break  
  
    print(f"Final transaction graph: {G.number_of_nodes()} wallet addresses, {G.number_of_edges()} transactions")  
    
    # Analyze edge labels for debugging
    analyze_edge_labels(G)
      
    # Convert to the expected dictionary format
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }
      
    # Save as dictionary
    with open(output_pkl, 'wb') as f:  
        pickle.dump(graph_data, f)  
      
    print(f"✅ Saved transaction network to {output_pkl}")  
      
    return graph_data  

def analyze_edge_labels(G):
    """Analyze what edge labels we have for debugging"""
    print("\n🔍 Edge Label Analysis:")
    edge_labels = {}
    
    for edge in G.edges(data=True):
        label = edge[2].get('label', 'NO_LABEL')
        edge_type = edge[2].get('type', 'NO_TYPE')
        
        if label not in edge_labels:
            edge_labels[label] = 0
        edge_labels[label] += 1
    
    print("Edge labels found:")
    for label, count in edge_labels.items():
        print(f"  - '{label}': {count} edges")
    
    # Check if we have the right attributes
    sample_edge = list(G.edges(data=True))[0] if G.number_of_edges() > 0 else None
    if sample_edge:
        print(f"\nSample edge attributes: {sample_edge[2]}")

def verify_saved_graph(pkl_path):  
    """Verify that the saved pickle file has proper edge labels"""  
    print(f"\nVerifying {pkl_path}...")  
    with open(pkl_path, 'rb') as f:  
        data = pickle.load(f)  
        print(f"Type: {type(data)}")  
        if isinstance(data, dict) and 'edges' in data:
            print(f"Number of edges: {len(data['edges'])}")  
            if len(data['edges']) > 0:
                sample_edge = data['edges'][0]
                print(f"Sample edge: {sample_edge}")
                print(f"Edge attributes: {sample_edge[2]}")
                print(f"Edge has 'label': {'label' in sample_edge[2]}")
                print(f"Edge has 'type': {'type' in sample_edge[2]}")

# Usage  
if __name__ == "__main__":  
    convert_token_transfers_to_pkl(  
        csv_path='/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/script/data/token_transfers.csv',
        output_pkl='stablecoin_network.pkl',
        sample_size=50000  
    )