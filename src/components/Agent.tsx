import React, { FC, useEffect, useRef } from 'react';
import { Agent as AgentType } from 'types';

interface AgentProps {
  agent: AgentType;
  numMovesRemaining: number;
  showDetails: boolean;
}

const nodeSize = 10;

const Agent: FC<AgentProps> = ({ agent, numMovesRemaining, showDetails }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasSize = agent.map.nodes.length * nodeSize;

  useEffect(() => {
    if (!canvasRef.current) {
      return;
    }

    const context = canvasRef.current.getContext('2d')!;

    agent.map.nodes.forEach((column, x) => {
      column.forEach((node, y) => {
        if (agent.position.x === x && agent.position.y === y) {
          context.fillStyle = '#4a4072';
        } else if (x === agent.map.end.x && y === agent.map.end.y) {
          context.fillStyle = '#FE4365';
        } else if (node.isObstruction) {
          context.fillStyle = '#79BD9A';
        } else {
          context.fillStyle = '#CFF09E';
        }
        context.fillRect(x * nodeSize, y * nodeSize, nodeSize, nodeSize);
      });
    });
  }, [agent, numMovesRemaining]);

  return (
    <div className="flex flex-col justify-center items-center border-4 border-solid border-white">
      <div>
        <canvas ref={canvasRef} width={canvasSize} height={canvasSize}></canvas>
      </div>
      {showDetails && (
        <div
          className="flex flex-row w-full justify-between mb-4"
          style={{ fontSize: 8 }}
        >
          <AgentDetails title="Inputs" data={agent.labeledInputs} />
          <AgentDetails title="Fitness" data={agent.labeledScoringFactors} />
        </div>
      )}
    </div>
  );
};

interface AgentDetailsProps {
  title: string;
  data: Record<string, number>;
}

const AgentDetails: FC<AgentDetailsProps> = ({ data, title }) => (
  <div>
    <table>
      <thead>
        <tr>
          <th colSpan={2}>{title}</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(data).map(([label, value]) => (
          <tr key={label}>
            <td className="text-right" style={{ width: 70 }}>
              {label}:&nbsp;
            </td>
            <td style={{ width: 40 }}>
              {value.toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
                signDisplay: 'always',
              })}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export { Agent };
