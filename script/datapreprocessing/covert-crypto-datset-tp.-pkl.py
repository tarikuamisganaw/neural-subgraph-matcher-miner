import pandas as pd  
import networkx as nx  
import pickle  
import numpy as np

def convert_token_transfers_to_pkl(csv_path, output_pkl, sample_size=8000):
    """  
    Convert token_transfers.csv to optimized graph format with real addresses and rich metadata.
    Nodes use actual wallet addresses as IDs and include meaningful numeric features for GNNs.
    """
    print(f"Processing {csv_path} with optimized sample size {sample_size}...")  
      
    # Create directed graph for token transfers  
    G = nx.DiGraph()  
    processed = 0  
    
    # Only read essential columns to save memory and time
    usecols = ['from_address', 'to_address', 'value', 'contract_address']
    
    # Stream and process with smaller chunks
    chunk_size = 2000  
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, nrows=sample_size, usecols=usecols):  
        for _, row in chunk.iterrows():  
            from_addr = row.get('from_address')  
            to_addr = row.get('to_address')  
              
            if pd.notna(from_addr) and pd.notna(to_addr) and from_addr != to_addr:  
                # Convert to string and lowercase for consistency
                from_addr = str(from_addr).lower()
                to_addr = str(to_addr).lower()
                
                # Add nodes with actual address as ID and label
                if from_addr not in G:  
                    G.add_node(from_addr, 
                               label=from_addr[:8] + "...",  # Shortened for readability
                               full_address=from_addr,       # Store full address for reference
                               feature=[0.0, 0.0])          # Will update after building graph
                
                if to_addr not in G:  
                    G.add_node(to_addr, 
                               label=to_addr[:8] + "...", 
                               full_address=to_addr, 
                               feature=[0.0, 0.0])
                  
                token_name = 'unknown'
                if 'contract_address' in row and pd.notna(row['contract_address']):
                    token_contract = str(row['contract_address'])
                    token_mapping = {
                        '0xdac17f958d2ee523a2206206994597c13d831ec7': 'USDT',
                        '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': 'USDC', 
                        '0x6b175474e89094c44da98b954eedeac495271d0f': 'DAI'
                    }
                    token_name = token_mapping.get(token_contract.lower(), 'unknown')

                # Create edge attributes
                edge_attrs = {
                    'weight': 1.0,  
                    'label': f'{token_name}_transfer' if token_name != 'unknown' else 'token_transfer',
                    'type': token_name if token_name != 'unknown' else 'transaction',
                    'amount': float(row['value']) if 'value' in row and pd.notna(row['value']) else 0.0,
                    'token_contract': token_contract if token_name != 'unknown' else None,
                    'full_label': f"{token_name}_transfer ({row['value']})"  # For visualization
                }
                  
                G.add_edge(from_addr, to_addr, **edge_attrs)  
              
            processed += 1  
            if processed % 2000 == 0:  
                print(f"Processed {processed} transactions...")  
          
        if processed >= sample_size:  
            break  
  
    print(f" Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")  
    
    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes for cleaner graph")
    
    # === ADD ANCHOR NODES FOR GNN ===
    anchor1 = "ANCHOR_1"
    anchor2 = "ANCHOR_2"
    G.add_node(anchor1, label="ANCHOR_1", full_address=anchor1, feature=[1.0, 0.0], is_anchor=True)
    G.add_node(anchor2, label="ANCHOR_2", full_address=anchor2, feature=[0.0, 1.0], is_anchor=True)
    
    # Connect anchors to a central node for alignment
    if len(G.nodes()) > 2:
        degrees = dict(G.degree())
        central_node = max(degrees, key=degrees.get)
        G.add_edge(anchor1, central_node, label="anchor_edge", type="anchor", weight=0.0, full_label="Anchor → Central")
        G.add_edge(anchor2, central_node, label="anchor_edge", type="anchor", weight=0.0, full_label="Anchor → Central")
    else:
        first_node = list(G.nodes())[0]
        G.add_edge(anchor1, first_node, label="anchor_edge", type="anchor", weight=0.0, full_label="Anchor → Node")
        G.add_edge(anchor2, first_node, label="anchor_edge", type="anchor", weight=0.0, full_label="Anchor → Node")

    print(f"Added 2 anchor nodes for node-anchored GNN training.")
    
    # === UPDATE NODE FEATURES WITH MEANINGFUL VALUES ===
    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        # Normalize degrees to avoid large values
        max_degree = max(1, max(dict(G.degree()).values()))
        G.nodes[node]['feature'] = [
            in_degree / max_degree,   # normalized in-degree
            out_degree / max_degree   # normalized out-degree
        ]
    
    print("✅ Updated node features with normalized in/out degrees.")

    # Analyze final graph
    analyze_final_graph(G)
      
    # Save as dictionary
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }
      
    with open(output_pkl, 'wb') as f:  
        pickle.dump(graph_data, f)  
      
    print(f" Saved optimized network to {output_pkl}")  
   
    return graph_data  

def analyze_final_graph(G):
    """Analyze the final graph for optimization insights"""
    print("\n Final Graph Analysis:")
    print(f"   - Connected nodes: {G.number_of_nodes()}")
    print(f"   - Transactions: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        avg_connections = (sum(in_degrees) + sum(out_degrees)) / len(in_degrees)
        print(f"   - Average connections per node: {avg_connections:.2f}")
        
        # Edge types
        edge_labels = {}
        for edge in G.edges(data=True):
            label = edge[2].get('label', 'unknown')
            if label not in edge_labels:
                edge_labels[label] = 0
            edge_labels[label] += 1
        print(f"   - Edge types found: {edge_labels}")
        
        # Check connectivity
        components = nx.number_weakly_connected_components(G)
        if components == 1:
            print("   - Graph is fully connected: yes")
        else:
            print(f"   - Graph has {components} connected components")
            
        # Sample edges
        print(f"\n🔍 Sample edge verification:")
        sample_edges = list(G.edges(data=True))[:2]
        for i, (src, dst, attrs) in enumerate(sample_edges):
            print(f"   Edge {i+1}: {src[:8]}... → {dst[:8]}...")
            print(f"     Label: {attrs.get('label', 'N/A')}")
            print(f"     Amount: {attrs.get('amount', 'N/A')}")
            print(f"     Full label: {attrs.get('full_label', 'N/A')}")

def verify_saved_graph(pkl_path):  
    """Verify saved pickle file"""  
    print(f"\nVerifying {pkl_path}...")  
    with open(pkl_path, 'rb') as f:  
        data = pickle.load(f)  
        print(f"File type: {type(data)}")  
        if isinstance(data, dict) and 'edges' in data:
            print(f"Number of edges: {len(data['edges'])}")  
            if len(data['edges']) > 0:
                sample_edge = data['edges'][0]
                print(f"Sample edge attributes: {sample_edge[2]}")
                print(f"Edge has 'label': {'label' in sample_edge[2]}")
                print(f"Edge has 'full_label': {'full_label' in sample_edge[2]}")
                print(f"Edge has 'amount': {'amount' in sample_edge[2]}")

# Usage  
if __name__ == "__main__":  
    convert_token_transfers_to_pkl(  
        csv_path='/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/script/data/token_transfers.csv',
        output_pkl='stablecoin_network1.pkl',
        sample_size=8000  
    )