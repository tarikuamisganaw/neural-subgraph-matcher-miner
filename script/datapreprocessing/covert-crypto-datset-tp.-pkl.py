import zipfile  
import pandas as pd  
import networkx as nx  
import pickle  
import torch  
  
def convert_zip_to_pkl(zip_path, output_pkl, sample_size=50000):  
    """  
    Convert cryptocurrency transaction data from ZIP file to the EXPECTED dictionary format.  
    Creates a format compatible with the neural subgraph matcher-miner system.  
    """  
    print(f"Processing {zip_path}...")  
      
    # Create directed graph for cryptocurrency transactions  
    G = nx.DiGraph()  
    processed = 0  
      
    with zipfile.ZipFile(zip_path, 'r') as z:  
        # Find the main CSV file inside ZIP  
        csv_files = [name for name in z.namelist() if name.endswith('.csv')]  
        if not csv_files:  
            raise Exception("No CSV files found in ZIP")  
          
        main_csv = csv_files[0]  
        print(f"Found CSV: {main_csv}")  
          
        # Stream and process the CSV  
        with z.open(main_csv) as f:  
            chunk_size = 10000  
            for chunk in pd.read_csv(f, chunksize=chunk_size, nrows=sample_size):  
                for _, row in chunk.iterrows():  
                    from_addr = row.get('from_address') or row.get('from')  
                    to_addr = row.get('to_address') or row.get('to')  
                      
                    if from_addr and to_addr:  
                        # Add nodes with required attributes  
                        if from_addr not in G:  
                            G.add_node(  
                                from_addr,  
                                label='address',  
                                id=str(from_addr)
                            )  
                        if to_addr not in G:  
                            G.add_node(  
                                to_addr,  
                                label='address',  
                                id=str(to_addr)
                            )  
                          
                        # Add edge with attributes  
                        edge_attrs = {'weight': 1.0}  # Default weight  
                        if 'value' in row and pd.notna(row['value']):  
                            edge_attrs['value'] = float(row['value'])  
                        if 'token_symbol' in row and pd.notna(row['token_symbol']):  
                            edge_attrs['token'] = str(row['token_symbol'])  
                            edge_attrs['type'] = str(row['token_symbol'])  # For edge type visualization  
                          
                        G.add_edge(from_addr, to_addr, **edge_attrs)  
                      
                    processed += 1  
                    if processed % 10000 == 0:  
                        print(f"Processed {processed} transactions...")  
                  
                if processed >= sample_size:  
                    break  
      
    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")  
      
    # ✅ FIX: Convert to the EXPECTED dictionary format with 'nodes' and 'edges' keys
    graph_data = {
        'nodes': list(G.nodes(data=True)),  # This is what the code expects
        'edges': list(G.edges(data=True))   # This is what the code expects
    }
      
    # Save as dictionary (not as NetworkX graph)
    with open(output_pkl, 'wb') as f:  
        pickle.dump(graph_data, f)  
      
    print(f"Saved to {output_pkl}")  
      
    # Verify the saved file  
    verify_saved_graph(output_pkl)  
      
    return graph_data  
  
def verify_saved_graph(pkl_path):  
    """Verify that the saved pickle file is in the correct format"""  
    print(f"\nVerifying {pkl_path}...")  
    with open(pkl_path, 'rb') as f:  
        data = pickle.load(f)  
        print(f"Type: {type(data)}")  
        print(f"Is dict: {isinstance(data, dict)}")  
        if isinstance(data, dict):  
            print(f"Dict keys: {list(data.keys())}")  
            if 'nodes' in data:  
                print(f"Number of nodes: {len(data['nodes'])}")  
                if len(data['nodes']) > 0:  
                    print(f"Sample node: {data['nodes'][0]}")  
            if 'edges' in data:  
                print(f"Number of edges: {len(data['edges'])}")  
                if len(data['edges']) > 0:  
                    print(f"Sample edge: {data['edges'][0]}")  
        else:  
            print("WARNING: File is not a dictionary!")  
  
# Usage  
if __name__ == "__main__":  
    convert_zip_to_pkl(  
        zip_path='/home/tarik/Downloads/neurograph/neural-subgraph-matcher-miner/script/data/ERC20-stablecoins.zip',  
        output_pkl='stablecoin_sample.pkl',  # Save to repo root for workflow  
        sample_size=50000  
    )