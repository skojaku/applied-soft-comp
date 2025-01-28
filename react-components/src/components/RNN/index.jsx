import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const RNNVisualization = ({ Slider, Switch }) => {
  const [weight, setWeight] = useState(0.5);
  const [input, setInput] = useState(1.0);
  const [initialState, setInitialState] = useState(1.0);
  const [useTanh, setUseTanh] = useState(false);
  const [timeSeriesData, setTimeSeriesData] = useState([]);

  const HIDDEN_COLOR = "#6B4CE6";
  const INPUT_COLOR = "#1D84B5";
  const OUTPUT_COLOR = "#E6425E";

  useEffect(() => {
    let data = [];
    let state = initialState;

    for (let t = 0; t < 50; t++) {
      const currentState = useTanh ?
        Math.tanh(weight * state + input) :
        weight * state + input;

      data.push({
        time: t,
        input: input,
        output: currentState
      });
      state = currentState;
    }

    setTimeSeriesData(data);
  }, [weight, input, initialState, useTanh]);

  const SchematicSVG = () => (
    <svg viewBox="0 0 200 120" style={{ width: '100%', height: '120px' }}>
      <rect x="70" y="30" width="60" height="40" fill="white" stroke={HIDDEN_COLOR} strokeWidth="2"/>
      <text x="100" y="50" textAnchor="middle" dominantBaseline="middle" style={{ fontSize: '12px' }}>
        {useTanh ? "tanh(h)" : "h"}
      </text>

      <path
        d="M100 30 C100 0, 150 0, 150 30 L150 70 C150 100, 100 100, 100 70"
        fill="none"
        stroke={HIDDEN_COLOR}
        strokeWidth={Math.max(Math.abs(weight) * 6, 0.5)}
      />

      <line x1="40" y1="50" x2="70" y2="50" stroke={INPUT_COLOR} strokeWidth="2" markerEnd="url(#arrowhead-input)"/>
      <text x="30" y="50" textAnchor="middle" style={{ fontSize: '12px' }}>u</text>

      <line x1="130" y1="50" x2="160" y2="50" stroke={OUTPUT_COLOR} strokeWidth="2" markerEnd="url(#arrowhead-output)"/>
      <text x="170" y="50" textAnchor="middle" style={{ fontSize: '12px' }}>y</text>

      <defs>
        <marker
          id="arrowhead-input"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={INPUT_COLOR}/>
        </marker>
        <marker
          id="arrowhead-output"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={OUTPUT_COLOR}/>
        </marker>
      </defs>

      <text x="100" y="15" textAnchor="middle" style={{ fontSize: '12px' }}>w = {weight.toFixed(2)}</text>
    </svg>
  );

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: '500', marginBottom: '0.5rem' }}>Initial State h(0): {initialState.toFixed(2)}</h3>
        <Slider.Root
          value={[initialState]}
          onValueChange={([value]) => setInitialState(value)}
          min={-2}
          max={2}
          step={0.1}
        />
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: '500', marginBottom: '0.5rem' }}>Recurrent Weight: {weight.toFixed(2)}</h3>
        <Slider.Root
          value={[weight]}
          onValueChange={([value]) => setWeight(value)}
          min={-2}
          max={2}
          step={0.1}
        />
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h3 style={{ fontSize: '1.25rem', fontWeight: '500', marginBottom: '0.5rem' }}>Input Signal: {input.toFixed(2)}</h3>
        <Slider.Root
          value={[input]}
          onValueChange={([value]) => setInput(value)}
          min={-2}
          max={2}
          step={0.1}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
        <Switch.Root
          checked={useTanh}
          onCheckedChange={setUseTanh}
        />
        <label style={{ fontSize: '1.25rem', fontWeight: '500' }}>Use tanh non-linearity</label>
      </div>

      <SchematicSVG />

      <div style={{ marginTop: '20px' }}>
        <LineChart
          width={600}
          height={200}
          data={timeSeriesData}
          margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="input" stroke={INPUT_COLOR} dot={false} name="Input" strokeWidth={2}/>
          <Line type="monotone" dataKey="output" stroke={OUTPUT_COLOR} dot={false} name="Output" strokeWidth={2}/>
        </LineChart>
      </div>
    </div>
  );
};

export default RNNVisualization;