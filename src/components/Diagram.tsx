import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  // DeepMind-style colors: Blue/Grey/White
  themeVariables: { 
    primaryColor: '#e0eaff', 
    lineColor: '#2563eb',
    primaryTextColor: '#1e293b',
    primaryBorderColor: '#2563eb'
  }
});

export default function Diagram({ code }: { code: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      mermaid.contentLoaded();
    }
  }, [code]);

  return (
    <div className="mermaid flex justify-center p-6 bg-slate-50 border border-slate-200 rounded-lg my-8 shadow-sm">
      {code}
    </div>
  );
}