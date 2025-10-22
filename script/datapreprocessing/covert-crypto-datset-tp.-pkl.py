import pandas as pd  
import networkx as nx  
import pickle  
import numpy as np

def convert_token_transfers_to_pkl(csv_path, output_pkl, sample_size=8000):
    """  
    Convert token_transfers.csv to a clean graph format for subgraph mining.
    
    - Node labels: shortened wallet addresses (e.g., '0xd30b...')
    - Edge labels: 'USDT_transfer (18.67)' for interpretability
    - Anchor nodes added for node-anchored GNN training
    """
    print(f"Processing {csv_path} with optimized sample size {sample_size}...")  
      
    # Create directed graph
    G = nx.DiGraph()  
    processed = 0  
    
    # Read only essential columns
    usecols = ['from_address', 'to_address', 'value', 'contract_address']
    chunk_size = 2000  # Process in small chunks for memory efficiency
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, nrows=sample_size, usecols=usecols):  
        for _, row in chunk.iterrows():  
            from_addr = row.get('from_address')  
            to_addr = row.get('to_address')  
              
            # Skip invalid or self-transfers
            if pd.notna(from_addr) and pd.notna(to_addr) and from_addr != to_addr:  
                from_addr = str(from_addr).lower()
                to_addr = str(to_addr).lower()
                
                # Add nodes with CLEAN label = shortened address
                if from_addr not in G:  
                    G.add_node(from_addr, 
                               label=from_addr[:8] + "...",      #  Shown in visualization
                               feature=[0.0, 0.0])              
                
                if to_addr not in G:  
                    G.add_node(to_addr, 
                               label=to_addr[:8] + "...", 
                               feature=[0.0, 0.0])
                  
                # Map token contract to name
                token_name = 'unknown'
                if 'contract_address' in row and pd.notna(row['contract_address']):
                    token_contract = str(row['contract_address']).lower()
                    token_mapping = {
                        '0xdac17f958d2ee523a2206206994597c13d831ec7': 'USDT',
                        '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48': 'USDC', 
                        '0x6b175474e89094c44da98b954eedeac495271d0f': 'DAI'
                    }
                    token_name = token_mapping.get(token_contract, 'unknown')

                # Parse amount
                amount = float(row['value']) if 'value' in row and pd.notna(row['value']) else 0.0

                # Create edge label: "USDT_transfer (18.67)"
                edge_label = f"{token_name}_transfer ({amount})" if token_name != 'unknown' else f"transfer ({amount})"

                # Edge attributes
                edge_attrs = {
                    'weight': 1.0,                    # Unbiased weight for GNN
                    'label': edge_label,              # Shown on edges in visualization
                    'type': token_name,
                    'amount': amount,
                    'token_contract': row.get('contract_address')
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
    
    # Add anchor nodes (required if using --node_anchored True)
    anchor1, anchor2 = "ANCHOR_1", "ANCHOR_2"
    G.add_node(anchor1, label="ANCHOR_1", feature=[1.0, 0.0])
    G.add_node(anchor2, label="ANCHOR_2", feature=[0.0, 1.0])
    
    # Connect anchors to a central node for alignment
    if len(G.nodes()) > 2:
        degrees = dict(G.degree())
        central_node = max(degrees, key=degrees.get)
        G.add_edge(anchor1, central_node, label="anchor", weight=0.0)
        G.add_edge(anchor2, central_node, label="anchor", weight=0.0)
    else:
        if len(G.nodes()) > 0:
            first_node = list(G.nodes())[0]
            G.add_edge(anchor1, first_node, label="anchor", weight=0.0)
            G.add_edge(anchor2, first_node, label="anchor", weight=0.0)

    # Update node features with normalized in/out degree (for GNN only)
    if G.number_of_nodes() > 0:
        max_degree = max(dict(G.degree()).values())
        max_degree = max(max_degree, 1)  # Avoid division by zero
        for node in G.nodes():
            in_deg = G.in_degree(node) / max_degree
            out_deg = G.out_degree(node) / max_degree
            G.nodes[node]['feature'] = [in_deg, out_deg]
        print("Node features updated with normalized degrees (for GNN training).")
    else:
        print("Graph has no nodes after cleaning.")

    # Save as dictionary
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }
    
    with open(output_pkl, 'wb') as f:  
        pickle.dump(graph_data, f)  
      
    print(f"Saved optimized network to {output_pkl}")  
    return graph_data  

#verification function
def verify_saved_graph(pkl_path):  
    print(f"\n Verifying {pkl_path}...")  
    with open(pkl_path, 'rb') as f:  
        data = pickle.load(f)  
        print(f"Type: {type(data)}")  
        if isinstance(data, dict) and 'nodes' in data and 'edges' in data:
            print(f"Nodes: {len(data['nodes'])}, Edges: {len(data['edges'])}")
            if data['nodes']:
                node_id, attrs = data['nodes'][0]
                print(f"Sample node ID: {node_id}")
                print(f"Node label (for viz): {attrs.get('label', 'N/A')}")
                print(f"Node feature (for GNN): {attrs.get('feature', 'N/A')}")
            if data['edges']:
                u, v, attrs = data['edges'][0]
                print(f"Sample edge label: {attrs.get('label', 'N/A')}")

# Run the conversion
if __name__ == "__main__":  
    convert_token_transfers_to_pkl(  
        csv_path='/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/script/data/token_transfers.csv',
        output_pkl='stablecoin_network1.pkl',
        sample_size=8000  
    )
    #verify
    verify_saved_graph('stablecoin_network1.pkl')