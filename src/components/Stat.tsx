import React, { FC, ReactNode } from 'react';

interface StatProps {
  label: string;
  value: ReactNode;
}

const Stat: FC<StatProps> = ({ label, value }) => {
  return (
    <div className="text-center text-xs mx-6 mb-2">
      <div className="font-semibold">{label}</div>
      <div>{value}</div>
    </div>
  );
};

export { Stat };
