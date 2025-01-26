import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import * as Slider from '@radix-ui/react-slider';
import * as Switch from '@radix-ui/react-switch';

const RNNVisualization = () => {
  const [weight, setWeight] = useState(0.5);
  const [input, setInput] = useState(1.0);
  const [useTanh, setUseTanh] = useState(false);
  const [timeSeriesData, setTimeSeriesData] = useState([]);

  const HIDDEN_COLOR = "#E6425E";
  const INPUT_COLOR = "#1D84B5";

  useEffect(() => {
    let data = [];
    let state = 1;

    for (let t = 0; t < 50; t++) {
      data.push({
        time: t,
        value: state,
        input: input
      });
      state = useTanh ?
        Math.tanh(weight * state + input) :
        weight * state + input;
    }

    setTimeSeriesData(data);
  }, [weight, input, useTanh]);

  const SchematicSVG = () => (
    <svg viewBox="0 0 200 120" className="w-full h-32">
      <rect x="70" y="30" width="60" height="40" fill="white" stroke={HIDDEN_COLOR} strokeWidth="2"/>
      <text x="100" y="50" textAnchor="middle" dominantBaseline="middle" className="text-sm">
        {useTanh ? "tanh(h)" : "h"}
      </text>

      <path
        d="M100 30 C100 10, 100 10, 100 10 L100 10 C100 10, 100 30, 100 30"
        fill="none"
        stroke={HIDDEN_COLOR}
        strokeWidth="2"
      />

      <line x1="40" y1="50" x2="70" y2="50" stroke={INPUT_COLOR} strokeWidth="2" markerEnd="url(#arrowhead)"/>
      <text x="30" y="50" textAnchor="middle" className="text-xs">u</text>

      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={INPUT_COLOR}/>
        </marker>
      </defs>

      <text x="100" y="15" textAnchor="middle" className="text-xs">w = {weight.toFixed(2)}</text>
    </svg>
  );

  return (
    <div className="p-6 space-y-6 bg-white rounded-lg shadow-md">
      <div className="prose max-w-none mb-4">
        <h2 className="text-xl font-bold mb-2">Simple RNN</h2>
        <p>Equation: {useTanh ? 'h(t) = tanh(w * h(t-1) + u)' : 'h(t) = w * h(t-1) + u'}</p>
        <p>System behavior: |w| &lt; 1 stable, |w| &gt; 1 unstable {useTanh && ', tanh bounds output to [-1,1]'}</p>
      </div>

      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-medium mb-2">Recurrent Weight: {weight.toFixed(2)}</h3>
          <Slider.Root
            value={[weight]}
            onValueChange={([value]) => setWeight(value)}
            min={-1.5}
            max={1.5}
            step={0.01}
            className="slider-root"
          >
            <Slider.Track className="slider-track">
              <Slider.Range className="slider-range" />
            </Slider.Track>
            <Slider.Thumb className="slider-thumb" />
          </Slider.Root>
        </div>

        <div>
          <h3 className="text-lg font-medium mb-2">Input Signal: {input.toFixed(2)}</h3>
          <Slider.Root
            value={[input]}
            onValueChange={([value]) => setInput(value)}
            min={-2}
            max={2}
            step={0.1}
            className="slider-root"
          >
            <Slider.Track className="slider-track">
              <Slider.Range className="slider-range" />
            </Slider.Track>
            <Slider.Thumb className="slider-thumb" />
          </Slider.Root>
        </div>

        <div className="flex items-center space-x-2">
          <Switch.Root
            checked={useTanh}
            onCheckedChange={setUseTanh}
            className="switch-root"
          >
            <Switch.Thumb className="switch-thumb" />
          </Switch.Root>
          <label className="text-lg font-medium">Use tanh non-linearity</label>
        </div>
      </div>

      <SchematicSVG />

      <div className="mt-6 h-64">
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
          <Line type="monotone" dataKey="value" stroke={HIDDEN_COLOR} dot={false} name="Hidden State" strokeWidth={2}/>
          <Line type="monotone" dataKey="input" stroke={INPUT_COLOR} dot={false} name="Input" strokeWidth={2}/>
        </LineChart>
      </div>
    </div>
  );
};

export default RNNVisualization;