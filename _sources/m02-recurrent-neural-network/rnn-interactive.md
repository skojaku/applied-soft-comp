# Interactive RNN Visualization

This interactive visualization demonstrates the behavior of a simple Recurrent Neural Network (RNN). You can experiment with different parameters to understand how they affect the network's behavior.

<div class="rnn-visualization-container"></div>

<script type="text/javascript">
window.addEventListener('load', function() {
    if (window.ReactComponents && window.ReactComponents.RNNVisualization) {
        const containers = document.querySelectorAll('.rnn-visualization-container');
        containers.forEach(container => {
            ReactDOM.render(
                React.createElement(window.ReactComponents.RNNVisualization),
                container
            );
        });
    } else {
        console.error('React components not loaded properly');
    }
});
</script>

## Understanding the Visualization

- **Recurrent Weight (w)**: Controls how much the previous state influences the current state
- **Input Signal (u)**: The external input to the system
- **Tanh Non-linearity**: Toggle to see how the tanh activation function affects the system's behavior

The graph shows how the hidden state evolves over time based on the current parameters.