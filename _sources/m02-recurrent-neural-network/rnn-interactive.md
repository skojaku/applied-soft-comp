# Interactive RNN Visualization

This interactive visualization demonstrates the behavior of a simple Recurrent Neural Network (RNN). You can experiment with different parameters to understand how they affect the network's behavior.

<div class="rnn-visualization-container"></div>

<script src="https://unpkg.com/react@17/umd/react.development.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js" crossorigin></script>
<script src="https://unpkg.com/recharts@2.10.3/umd/Recharts.js" crossorigin></script>
<script src="https://unpkg.com/@radix-ui/primitive@1.0.1/dist/index.umd.js" crossorigin></script>

<style>
.rnn-visualization-container {
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 20px 0;
}

.recharts-wrapper {
    margin: 0 auto;
}

.slider-root {
    position: relative;
    display: flex;
    align-items: center;
    width: 200px;
    height: 20px;
}

.slider-track {
    background-color: #E6E8EC;
    position: relative;
    flex-grow: 1;
    height: 3px;
}

.slider-range {
    position: absolute;
    background-color: #6B4CE6;
    height: 100%;
}

.slider-thumb {
    display: block;
    width: 20px;
    height: 20px;
    background-color: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-radius: 10px;
    cursor: pointer;
}

.switch-root {
    width: 42px;
    height: 25px;
    background-color: #E6E8EC;
    border-radius: 9999px;
    position: relative;
    cursor: pointer;
    border: none;
}

.switch-thumb {
    width: 21px;
    height: 21px;
    background-color: white;
    border-radius: 9999px;
    transition: transform 100ms;
    transform: translateX(2px);
    will-change: transform;
}

.switch-thumb[data-state='checked'] {
    transform: translateX(19px);
}

.switch-root[data-state='checked'] {
    background-color: #6B4CE6;
}
</style>

<script>
window.process = {
    env: {
        NODE_ENV: 'production'
    }
};

// Define custom slider and switch components
window.CustomSlider = {
    Root: function(props) {
        return React.createElement('div', {
            className: 'slider-root',
            onClick: function(e) {
                const rect = e.currentTarget.getBoundingClientRect();
                const pos = (e.clientX - rect.left) / rect.width;
                const value = props.min + (props.max - props.min) * pos;
                props.onValueChange([Math.min(Math.max(value, props.min), props.max)]);
            }
        }, [
            React.createElement('div', { className: 'slider-track', key: 'track' }, [
                React.createElement('div', {
                    className: 'slider-range',
                    key: 'range',
                    style: { width: ((props.value[0] - props.min) / (props.max - props.min) * 100) + '%' }
                })
            ]),
            React.createElement('div', {
                className: 'slider-thumb',
                key: 'thumb',
                style: { left: ((props.value[0] - props.min) / (props.max - props.min) * 100) + '%' }
            })
        ]);
    }
};

window.CustomSwitch = {
    Root: function(props) {
        return React.createElement('button', {
            className: 'switch-root',
            'data-state': props.checked ? 'checked' : 'unchecked',
            onClick: function() { props.onCheckedChange(!props.checked); }
        }, [
            React.createElement('span', {
                className: 'switch-thumb',
                key: 'thumb',
                'data-state': props.checked ? 'checked' : 'unchecked'
            })
        ]);
    }
};
</script>

<script src="../_static/react-components.umd.js"></script>

<script type="text/javascript">
document.addEventListener('DOMContentLoaded', function() {
    const container = document.querySelector('.rnn-visualization-container');
    if (container && window.React && window.ReactDOM && window.ReactComponents?.RNNVisualization) {
        ReactDOM.render(
            React.createElement(window.ReactComponents.RNNVisualization, {
                Slider: window.CustomSlider,
                Switch: window.CustomSwitch
            }),
            container
        );
    } else {
        console.error('Failed to initialize RNN visualization:', {
            container: !!container,
            React: !!window.React,
            ReactDOM: !!window.ReactDOM,
            RNNVisualization: !!window.ReactComponents?.RNNVisualization
        });
    }
});
</script>

## Understanding the Visualization

- **Recurrent Weight (w)**: Controls how much the previous state influences the current state
  - Values < 1: System converges to a stable point
  - Values > 1: System becomes unstable and grows exponentially
  - Values < 0: System oscillates

- **Input Signal (u)**: The external input to the system
  - Positive values shift the system upward
  - Negative values shift the system downward
  - Zero means no external input

- **Tanh Non-linearity**: Toggle to see how the tanh activation function affects the system's behavior
  - When enabled, bounds the output between -1 and 1
  - Helps prevent exploding values
  - Makes the system more stable

The graph shows how the hidden state evolves over time based on the current parameters. The red line represents the hidden state, while the blue line shows the input signal.