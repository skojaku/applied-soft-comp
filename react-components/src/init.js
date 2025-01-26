import React from 'react';
import ReactDOM from 'react-dom';
import { RNNVisualization } from './components';

// Initialize all RNN visualizations on the page
document.addEventListener('DOMContentLoaded', () => {
  const containers = document.querySelectorAll('.rnn-visualization-container');
  containers.forEach(container => {
    ReactDOM.render(
      React.createElement(RNNVisualization),
      container
    );
  });
});