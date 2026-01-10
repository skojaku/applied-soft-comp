import sys
import os

# Add the current directory to path
sys.path.append(os.getcwd())

try:
    import transformer_utils
    import transformer_viz
    import transformer

    print("Imports successful")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

try:
    transformer_viz.visualize_static_embeddings()
    transformer_viz.visualize_contextualization()
    transformer_viz.visualize_attention_mechanism()
    print("Visualization functions executed successfully (returned UI objects)")
except Exception as e:
    print(f"Visualization execution failed: {e}")
    sys.exit(1)
