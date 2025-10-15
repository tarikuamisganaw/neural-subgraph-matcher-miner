import pandas as pd  
import networkx as nx  
import pickle  
  
def csv_to_graph(csv_path, output_pkl, sample_size=50000):
    """Simple converter from token_transfers.csv to graph format"""
    print(f"Reading {csv_path}...")
    
    G = nx.DiGraph()
    df = pd.read_csv(csv_path, nrows=sample_size)
    
    for _, row in df.iterrows():
        from_addr = str(row['from_address']).lower() if pd.notna(row['from_address']) else None
        to_addr = str(row['to_address']).lower() if pd.notna(row['to_address']) else None
        
        if from_addr and to_addr and from_addr != to_addr:
            # Add nodes
            if from_addr not in G:
                G.add_node(from_addr, label='address')
            if to_addr not in G:
                G.add_node(to_addr, label='address')
            
            # Add edge
            edge_attrs = {'weight': 1.0}
            if pd.notna(row.get('value')):
                edge_attrs['amount'] = float(row['value'])
            
            G.add_edge(from_addr, to_addr, **edge_attrs)
    
    print(f"Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Convert to required format
    graph_data = {
        'nodes': list(G.nodes(data=True)),
        'edges': list(G.edges(data=True))
    }
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(graph_data, f)
    
    print(f"✅ Saved to {output_pkl}")
    return graph_data

# Run it
if __name__ == "__main__":
    csv_to_graph(
        csv_path='/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/script/data/token_transfers.csv',
        output_pkl='stablecoin_network.pkl',
        sample_size=50000
    )