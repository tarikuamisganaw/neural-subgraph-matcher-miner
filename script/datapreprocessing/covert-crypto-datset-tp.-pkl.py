import zipfile
import pandas as pd
import networkx as nx
import pickle

def convert_zip_to_pkl(zip_path, output_pkl, sample_size=50000):
    """
    Convert directly from ZIP file without extracting
    """
    print(f"Processing {zip_path}...")
    
    G = nx.DiGraph()
    processed = 0
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the main CSV file inside ZIP
        csv_files = [name for name in z.namelist() if name.endswith('.csv')]
        if not csv_files:
            raise Exception("No CSV files found in ZIP")
        
        main_csv = csv_files[0]  # Use the first CSV found
        print(f"Found CSV: {main_csv}")
        
        # Stream and process the CSV
        with z.open(main_csv) as f:
            # Process in chunks to manage memory
            chunk_size = 10000
            for chunk in pd.read_csv(f, chunksize=chunk_size, nrows=sample_size):
                for _, row in chunk.iterrows():
                    from_addr = row.get('from_address') or row.get('from')
                    to_addr = row.get('to_address') or row.get('to')
                    
                    if from_addr and to_addr:
                        # Add nodes
                        if from_addr not in G:
                            G.add_node(from_addr, label='address', id=str(from_addr))
                        if to_addr not in G:
                            G.add_node(to_addr, label='address', id=str(to_addr))
                        
                        # Add edge with attributes
                        edge_attrs = {}
                        if 'value' in row and pd.notna(row['value']):
                            edge_attrs['value'] = float(row['value'])
                        if 'token_symbol' in row:
                            edge_attrs['token'] = str(row['token_symbol'])
                        
                        G.add_edge(from_addr, to_addr, **edge_attrs)
                    
                    processed += 1
                    if processed % 10000 == 0:
                        print(f"Processed {processed} transactions...")
                
                if processed >= sample_size:
                    break
    
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Save the graph
    with open(output_pkl, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"Saved to {output_pkl}")
    return G

# Usage - this won't extract the full ZIP!
convert_zip_to_pkl('/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/script/data/ERC20-stablecoins.zip', '/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/stablecoin_sample.pkl', 50000)