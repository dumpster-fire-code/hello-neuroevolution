import { Engine } from 'engine/Engine';
import React, {
  createContext,
  FC,
  memo,
  ReactNode,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';

const updateInterval = 0;

const NeuroevolutionContext = createContext<{ engine: Engine }>({
  engine: undefined as never,
});

interface NeuroevolutionProviderProps {
  children: ReactNode;
}

const NeuroevolutionProvider: FC<NeuroevolutionProviderProps> = memo(
  ({ children }) => {
    const hasInitializedEngineRef = useRef(false);
    const [engine, setEngine] = useState<Engine>();
    const [, setLastUpdatedAt] = useState(0);

    useEffect(() => {
      if (!hasInitializedEngineRef.current) {
        setEngine(new Engine());
        hasInitializedEngineRef.current = true;
      }
    }, []);

    useEffect(() => {
      let timeout: ReturnType<typeof setTimeout>;

      if (engine) {
        const update = () => {
          if (engine.numMovesRemaining === 0) {
            engine.nextGeneration();
          } else {
            engine.update();
            setLastUpdatedAt(Date.now());
          }
          timeout = setTimeout(update, updateInterval);
        };

        timeout = setTimeout(update, updateInterval);
      }

      return () => clearTimeout(timeout);
    }, [engine]);

    if (!engine) {
      return null;
    }

    return (
      <NeuroevolutionContext.Provider value={{ engine }}>
        {children}
      </NeuroevolutionContext.Provider>
    );
  },
);

const useNeuroevolution = () => useContext(NeuroevolutionContext);

export { NeuroevolutionProvider, useNeuroevolution };
