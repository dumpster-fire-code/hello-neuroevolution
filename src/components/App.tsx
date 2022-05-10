import { NeuroevolutionProvider } from 'contexts/NeuroevolutionProvider';
import React, { FC } from 'react';

import { Dashboard } from './Dashboard';

const App: FC = () => {
  return (
    <NeuroevolutionProvider>
      <Dashboard />
    </NeuroevolutionProvider>
  );
};

export { App };
