import pandas as pd  
import networkx as nx  
import pickle  
import numpy as np

def convert_token_transfers_to_pkl(csv_path, output_pkl, sample_size=8000):  
    """  
    Convert token_transfers.csv to optimized graph format for faster processing.  
    Nodes use actual addresses as IDs and include numeric features for GNNs.
    """  
    print(f"Processing {csv_path} with optimized sample size {sample_size}...")  
      
    # Create directed graph for token transfers  
    G = nx.DiGraph()  
    processed = 0  
    
    # Only read essential columns to save memory and time
    usecols = ['from_address', 'to_address', 'value', 'contract_address']
    
    # Stream and process with smaller chunks
    chunk_size = 2000  # Smaller chunks for memory efficiency
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, nrows=sample_size, usecols=usecols):  
        for _, row in chunk.iterrows():  
            from_addr = row.get('from_address')  
            to_addr = row.get('to_address')  
              
            if pd.notna(from_addr) and pd.notna(to_addr) and from_addr != to_addr:  
                # Convert to string to ensure consistency
                from_addr = str(from_addr).lower()
                to_addr = str(to_addr).lower()
                
                # Add nodes with minimal attributes + numeric feature [0,0] for regular nodes
                if from_addr not in G:  
                    G.add_node(from_addr, label='address', feature=[0.0, 0.0])  
                if to_addr not in G:  
                    G.add_node(to_addr, label='address', feature=[0.0, 0.0])  
                  
                token_name = 'unknown'
                if 'contract_address' in row and pd.notna(row['contract_address']):
                    token_contract = str(row['contract_address'])
                    token_mapping = {
                        '0xdac17f958d2ee523a2206206994597c13d831ec7': 'USDT',
                        '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': 'USDC', 
                        '0x6b175474e89094c44da98b954eedeac495271d0f': 'DAI'
                    }
                    token_name = token_mapping.get(token_contract.lower(), 'unknown')

                # Create edge attributes with CORRECT weight (not transaction value)
                edge_attrs = {
                    'weight': 1.0,  
                    'label': f'{token_name}_transfer' if token_name != 'unknown' else 'token_transfer',
                    'type': token_name if token_name != 'unknown' else 'transaction'
                }
                
                # Add transaction amount 
                if 'value' in row and pd.notna(row['value']):  
                    amount = float(row['value'])
                    edge_attrs['amount'] = amount  # Store amount separately
                    # Weight stays as 1.0 - no bias toward large transactions
                    
                # Add token contract for reference
                if token_name != 'unknown':
                    edge_attrs['token_contract'] = token_contract
                  
                G.add_edge(from_addr, to_addr, **edge_attrs)  
              
            processed += 1  
            if processed % 2000 == 0:  
                print(f"Processed {processed} transactions...")  
          
        if processed >= sample_size:  
            break  
  
    print(f" Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")  
    
    # Remove isolated nodes to create cleaner, faster-to-process graph
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes for cleaner graph")
    
    # === ADD ANCHOR NODES HERE ===
    anchor1 = "ANCHOR_1"
    anchor2 = "ANCHOR_2"
    
    # Add anchor nodes with special features
    G.add_node(anchor1, label="anchor", feature=[1.0, 0.0], is_anchor=True)
    G.add_node(anchor2, label="anchor", feature=[0.0, 1.0], is_anchor=True)
    
    # Connect anchors to a central node (for alignment)
    if len(G.nodes()) > 2:  # Ensure there are non-anchor nodes
        # Pick the node with highest degree as "central"
        degrees = dict(G.degree())
        central_node = max(degrees, key=degrees.get)
        G.add_edge(anchor1, central_node, label="anchor_edge", type="anchor", weight=0.0)
        G.add_edge(anchor2, central_node, label="anchor_edge", type="anchor", weight=0.0)
    else:
        # Fallback: connect to first node
        first_node = list(G.nodes())[0]
        G.add_edge(anchor1, first_node, label="anchor_edge", type="anchor", weight=0.0)
        G.add_edge(anchor2, first_node, label="anchor_edge", type="anchor", weight=0.0)
    
    print(f"Added 2 anchor nodes for node-anchored GNN training.")
    
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
      
    print(f" Saved optimized network to {output_pkl}")  
   
    return graph_data  

def analyze_final_graph(G):
    """Analyze the final graph for optimization insights"""
    print("\n Final Graph Analysis:")
    print(f"   - Connected nodes: {G.number_of_nodes()}")
    print(f"   - Transactions: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        # Calculate basic graph metrics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        
        avg_connections = (sum(in_degrees) + sum(out_degrees)) / len(in_degrees)
        print(f"   - Average connections per node: {avg_connections:.2f}")
        
        # Analyze edge labels
        edge_labels = {}
        for edge in G.edges(data=True):
            label = edge[2].get('label', 'unknown')
            if label not in edge_labels:
                edge_labels[label] = 0
            edge_labels[label] += 1
        
        print(f"   - Edge types found: {edge_labels}")
        
        # Check graph connectivity
        components = nx.number_weakly_connected_components(G)
        if components == 1:
            print("   - Graph is fully connected: yes")
        else:
            print(f"   - Graph has {components} connected components")
            
        # Sample some edges to verify our fixes
        print(f"\n🔍 Sample edge verification:")
        sample_edges = list(G.edges(data=True))[:2]
        for i, (src, dst, attrs) in enumerate(sample_edges):
            print(f"   Edge {i+1}: {src[:8]}... → {dst[:8]}...")
            print(f"     Label: {attrs.get('label', 'N/A')}")
            print(f"     Weight: {attrs.get('weight', 'N/A')}")
            print(f"     Amount: {attrs.get('amount', 'N/A')}")
        
        # Check node features
        print(f"\n🔍 Node feature verification:")
        nodes_with_features = 0
        anchor_count = 0
        for node, attrs in G.nodes(data=True):
            if 'feature' in attrs:
                nodes_with_features += 1
                if attrs.get('label') == 'anchor':
                    anchor_count += 1
        print(f"   - Nodes with 'feature': {nodes_with_features}/{G.number_of_nodes()}")
        print(f"   - Anchor nodes detected: {anchor_count}")

def verify_saved_graph(pkl_path):  
    """Verify that the saved pickle file has proper edge labels and node features"""  
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
                print(f"Edge has 'type': {'type' in sample_edge[2]}")
                print(f"Edge weight is 1.0: {sample_edge[2].get('weight') == 1.0}")
            
            if 'nodes' in data and len(data['nodes']) > 0:
                sample_node = data['nodes'][0]
                print(f"\nSample node: {sample_node[0]}")
                print(f"Node attributes: {sample_node[1]}")
                print(f"Node has 'feature': {'feature' in sample_node[1]}")
                if 'feature' in sample_node[1]:
                    print(f"Feature vector: {sample_node[1]['feature']}")

# Usage  
if __name__ == "__main__":  
    convert_token_transfers_to_pkl(  
        csv_path='/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/script/data/token_transfers.csv',
        output_pkl='stablecoin_network2.pkl',
        sample_size=8000  # Reduced for faster training
    )