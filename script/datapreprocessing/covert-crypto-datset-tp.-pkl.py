import pandas as pd  
import networkx as nx  
import pickle  
  
def convert_token_transfers_to_pkl(csv_path, output_pkl, sample_size=15000):  # Reduced from 50k to 15k
    """  
    Convert token_transfers.csv to optimized graph format for faster processing.  
    """  
    print(f"Processing {csv_path} with optimized sample size {sample_size}...")  
      
    # Create directed graph for token transfers  
    G = nx.DiGraph()  
    processed = 0  
    
    # Only read essential columns to save memory and time
    usecols = ['from_address', 'to_address', 'value', 'contract_address']
    
    # Stream and process with smaller chunks
    chunk_size = 5000  
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, nrows=sample_size, usecols=usecols):  
        for _, row in chunk.iterrows():  
            from_addr = row.get('from_address')  
            to_addr = row.get('to_address')  
              
            if pd.notna(from_addr) and pd.notna(to_addr) and from_addr != to_addr:  
                # Convert to string to ensure consistency
                from_addr = str(from_addr).lower()
                to_addr = str(to_addr).lower()
                
                # Add nodes with minimal attributes
                if from_addr not in G:  
                    G.add_node(from_addr, label='address')  
                if to_addr not in G:  
                    G.add_node(to_addr, label='address')  
                  
                # Essential edge attributes for visualization
                edge_attrs = {
                    'weight': 1.0,
                    'label': 'token_transfer',  # Required for visualization
                    'type': 'transaction'       # Required for visualization
                }  
                
                # Add value if available
                if 'value' in row and pd.notna(row['value']):  
                    amount = float(row['value'])
                    edge_attrs['amount'] = amount
                    edge_attrs['weight'] = amount  # Use amount as weight
                    
                # Add token information if available
                if 'contract_address' in row and pd.notna(row['contract_address']):  
                    token_contract = str(row['contract_address'])
                    edge_attrs['token'] = token_contract
                    
                    # Map common tokens to readable names
                    token_mapping = {
                        '0xdac17f958d2ee523a2206206994597c13d831ec7': 'USDT',
                        '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': 'USDC', 
                        '0x6b175474e89094c44da98b954eedeac495271d0f': 'DAI'
                    }
                    
                    if token_contract.lower() in token_mapping:
                        token_name = token_mapping[token_contract.lower()]
                        edge_attrs['label'] = f'{token_name}_transfer'
                        edge_attrs['type'] = token_name
                  
                G.add_edge(from_addr, to_addr, **edge_attrs)  
              
            processed += 1  
            if processed % 5000 == 0:  
                print(f"Processed {processed} transactions...")  
          
        if processed >= sample_size:  
            break  
  
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")  
    
    # Remove isolated nodes to create cleaner, faster-to-process graph
    initial_nodes = G.number_of_nodes()
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes for cleaner graph")
    
    # Analyze the final graph
    analyze_final_graph(G)
      
    # Convert to the expected dictionary format
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }
      
    # Save as dictionary
    with open(output_pkl, 'wb') as f:  
        pickle.dump(graph_data, f)  
      
    print(f"✅ Saved optimized network to {output_pkl}")  
    print("🎯 This smaller dataset will run MUCH faster in your workflow!")
      
    return graph_data  

def analyze_final_graph(G):
    """Analyze the final graph for optimization insights"""
    print("\n📊 Final Graph Analysis:")
    print(f"   - Connected nodes: {G.number_of_nodes()}")
    print(f"   - Transactions: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        # Calculate basic graph metrics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        print(f"   - Average connections per node: {(sum(in_degrees) + sum(out_degrees)) / len(in_degrees):.2f}")
        print(f"   - Graph density: {nx.density(G):.4f}")
        
        # Check if graph is connected
        if nx.is_weakly_connected(G):
            print("   - Graph is connected: ✅")
        else:
            components = nx.number_weakly_connected_components(G)
            print(f"   - Graph has {components} connected components")
            
        # Analyze edge labels
        edge_labels = {}
        for edge in G.edges(data=True):
            label = edge[2].get('label', 'unknown')
            if label not in edge_labels:
                edge_labels[label] = 0
            edge_labels[label] += 1
        
        print(f"   - Edge types: {edge_labels}")

def verify_saved_graph(pkl_path):  
    """Verify that the saved pickle file has proper edge labels"""  
    print(f"\n🔍 Verifying {pkl_path}...")  
    with open(pkl_path, 'rb') as f:  
        data = pickle.load(f)  
        print(f"Type: {type(data)}")  
        if isinstance(data, dict) and 'edges' in data:
            print(f"Number of edges: {len(data['edges'])}")  
            if len(data['edges']) > 0:
                sample_edge = data['edges'][0]
                print(f"Sample edge attributes: {sample_edge[2]}")
                print(f"✅ Edge has 'label': {'label' in sample_edge[2]}")
                print(f"✅ Edge has 'type': {'type' in sample_edge[2]}")

# Usage  
if __name__ == "__main__":  
    convert_token_transfers_to_pkl(  
        csv_path='/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/script/data/token_transfers.csv',
        output_pkl='stablecoin_network.pkl',
        sample_size=15000  # Optimized for speed
    )