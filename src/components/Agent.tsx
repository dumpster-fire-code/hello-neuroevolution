import React, { FC, useEffect, useRef } from 'react';
import { Agent as AgentType } from 'types';

interface AgentProps {
  agent: AgentType;
  numMovesRemaining: number;
}

const nodeSize = 8;

const Agent: FC<AgentProps> = ({ agent, numMovesRemaining }) => {
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
    <div className="flex justify-center items-center border-4 border-solid border-white">
      <canvas ref={canvasRef} width={canvasSize} height={canvasSize}></canvas>
    </div>
  );
};

export { Agent };
