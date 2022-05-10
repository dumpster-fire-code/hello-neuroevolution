import { useNeuroevolution } from 'contexts/NeuroevolutionProvider';
import React, { FC, useState } from 'react';

import { Agent } from './Agent';
import { Stat } from './Stat';

const maxVisibleAgents = 32;

const formatNumber = (value: number) =>
  value.toLocaleString('en-US', { maximumFractionDigits: 4 });

const Dashboard: FC = () => {
  const {
    engine: {
      population,
      mutationRate,
      numGenerations,
      numMovesRemaining,
      populationSize,
    },
  } = useNeuroevolution();

  const [showAgents, setShowAgents] = useState(true);

  return (
    <div className="m-4">
      <div className="flex flex-row flex-wrap justify-center items-center">
        <Stat label="generation" value={formatNumber(numGenerations)} />
        <Stat label="population" value={formatNumber(populationSize)} />
        <Stat label="mutation_rate" value={formatNumber(mutationRate)} />
        <Stat label="moves_remaining" value={formatNumber(numMovesRemaining)} />
        <Stat
          label="show_agents"
          value={
            <input
              className="mt-1"
              type="checkbox"
              checked={showAgents}
              onChange={() => setShowAgents(!showAgents)}
            />
          }
        />
      </div>
      {showAgents && (
        <div className="flex flex-row flex-wrap justify-center">
          {population.slice(0, maxVisibleAgents).map((agent) => (
            <Agent
              key={agent.id}
              agent={agent}
              numMovesRemaining={numMovesRemaining}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export { Dashboard };
